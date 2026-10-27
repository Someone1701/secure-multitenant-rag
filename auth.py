import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("JWT_SECRET", "super-secure-multi-tenant-rag-key-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

security = HTTPBearer(auto_error=False)


def create_access_token(username: str, tenant_id: Optional[str] = None, expires_delta: Optional[timedelta] = None) -> str:
    """
    Issues a signed JWT containing username and an immutable tenant_id claim.
    If tenant_id is not explicitly provided, it is derived deterministically from the username.
    """
    clean_username = username.strip().lower()
    if not tenant_id:
        # Canonical tenant ID derivation from verified username
        tenant_id = f"tenant_{clean_username}"

    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    payload: Dict[str, Any] = {
        "sub": clean_username,
        "tenant_id": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> Dict[str, Any]:
    """
    Validates the bearer token and extracts tenant identity.
    Crucial constraint: Tenant ID is always derived from the cryptographic token,
    never from unverified client request parameters.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        tenant_id: Optional[str] = payload.get("tenant_id")
        username: Optional[str] = payload.get("sub")

        if not tenant_id or not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token claims: missing tenant_id or subject",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return {
            "tenant_id": tenant_id,
            "username": username,
            "claims": payload
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
