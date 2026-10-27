import os
import re
import json
import requests
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from auth import create_access_token, verify_token
from ingestion import (
    ingest_document,
    list_tenant_documents,
    delete_tenant_document,
    get_collection,
    compute_normalized_embeddings
)

app = FastAPI(
    title="Secure Multi-Tenant RAG System",
    description="Storage-Level Pre-Traversal Isolation, Deterministic Bounded Grounding, and Page-Coordinate Provenance Tracking",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Threshold parameter calibrated to tau = 0.65
RELEVANCE_THRESHOLD = 0.65
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


class LoginRequest(BaseModel):
    username: str


class QueryRequest(BaseModel):
    question: str
    threshold: Optional[float] = None


@app.post("/auth/login")
async def login(req: LoginRequest):
    """
    Accepts {username}, returns signed JWT with immutable tenant_id claim.
    Tenant identity is bound to this signed token.
    """
    username = req.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")

    token = create_access_token(username=username)
    tenant_id = f"tenant_{username.lower()}"
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": username,
        "tenant_id": tenant_id
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    auth_data: Dict[str, Any] = Depends(verify_token)
):
    """
    Accepts PDF or DOCX file.
    Tenant identity is extracted strictly from the cryptographic JWT, never from client body parameters.
    Performs layout-preserving extraction, sliding window chunking, S-BERT embedding, and ChromaDB storage.
    """
    tenant_id = auth_data["tenant_id"]

    filename = file.filename or "uploaded_document"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF (.pdf) or Word (.docx) file."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = ingest_document(
            file_bytes=file_bytes,
            filename=filename,
            tenant_id=tenant_id
        )
        return {
            "status": "success",
            "message": f"Document '{filename}' successfully ingested and isolated under tenant.",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/documents")
async def get_documents(auth_data: Dict[str, Any] = Depends(verify_token)):
    """
    Lists this tenant's uploaded documents.
    Storage-level isolation ensures no other tenant's documents are queried or returned.
    """
    tenant_id = auth_data["tenant_id"]
    documents = list_tenant_documents(tenant_id)
    return {
        "tenant_id": tenant_id,
        "count": len(documents),
        "documents": documents
    }


@app.delete("/documents/{doc_uuid}")
async def delete_document(
    doc_uuid: str,
    auth_data: Dict[str, Any] = Depends(verify_token)
):
    """
    Deletes a document and all associated vector chunks strictly belonging to this tenant.
    """
    tenant_id = auth_data["tenant_id"]
    deleted_count = delete_tenant_document(tenant_id, doc_uuid)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    return {
        "status": "success",
        "message": f"Successfully deleted document {doc_uuid} ({deleted_count} chunks removed)",
        "doc_uuid": doc_uuid,
        "deleted_chunks": deleted_count
    }


def get_ollama_model() -> str:
    """Detects available model in Ollama daemon, matching llama3 / llama3.2 if available."""
    configured = os.getenv("OLLAMA_MODEL")
    if configured:
        return configured
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            for m in models:
                if "llama3" in m:
                    return m
            if models:
                return models[0]
    except Exception:
        pass
    return "llama3"


def call_ollama(prompt: str, model: Optional[str] = None) -> str:
    """
    Dispatches generation to Ollama local API with detected or specified model.
    """
    active_model = model or get_ollama_model()
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": active_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 512
        }
    }
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code == 200:
        data = response.json()
        return data.get("response", "").strip()
    else:
        raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {response.text}")


@app.post("/query")
async def query_system(
    req: QueryRequest,
    auth_data: Dict[str, Any] = Depends(verify_token)
):
    """
    Core RAG Pipeline with Pre-Traversal Isolation, Deterministic Relevance Gate, and Provenance:
    1. Derive immutable tenant_id from token.
    2. Embed question with Sentence-BERT & L2-normalize.
    3. ChromaDB vector search with pre-traversal metadata constraint `where={"tenant_id": tenant_id}`.
    4. Deterministic Similarity Gate (tau = 0.65):
       If max(similarity) < 0.65 -> return fallback WITHOUT invoking LLM.
    5. Prompt construction enclosed in delimiter fences.
    6. LLM synthesis with exact page citation.
    """
    tenant_id = auth_data["tenant_id"]
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    collection = get_collection()

    # 1. Encode query and L2-normalize
    q_vec = compute_normalized_embeddings([question])[0]

    # 2. Storage-level tenant pre-traversal search (top 5 candidates)
    try:
        results = collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=5,
            where={"tenant_id": tenant_id},
            include=["documents", "metadatas", "embeddings", "distances"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    retrieved_embeddings = results.get("embeddings", [[]])[0]

    if docs is None or len(docs) == 0 or metadatas is None or len(metadatas) == 0:
        return {
            "answer": "Information not found in source material.",
            "citation": None,
            "gated": True,
            "s_max": 0.0,
            "threshold": RELEVANCE_THRESHOLD,
            "retrieved_candidates": 0,
            "reason": "No documents ingested for this tenant or no candidates match tenant predicate."
        }

    # 3. Calculate exact Cosine Similarities between q_vec and retrieved candidate vectors
    scores: List[float] = []
    if retrieved_embeddings is not None and len(retrieved_embeddings) > 0:
        for v in retrieved_embeddings:
            v_arr = np.array(v, dtype=np.float32)
            v_norm = np.linalg.norm(v_arr)
            if v_norm > 0:
                v_arr = v_arr / v_norm
            cos_sim = float(np.dot(q_vec, v_arr))
            scores.append(cos_sim)
    else:
        # Fallback to Chroma cosine distance conversion: sim = 1.0 - distance
        distances = results.get("distances", [[]])[0]
        scores = [max(0.0, min(1.0, 1.0 - d)) for d in distances]

    s_max = max(scores) if scores else 0.0
    active_threshold = req.threshold if req.threshold is not None else RELEVANCE_THRESHOLD

    # 4. Deterministic Relevance Gate (tau = 0.65)
    # If s_max < active_threshold, unconditionally terminate without calling the language model!
    if s_max < active_threshold:
        return {
            "answer": "Information not found in source material.",
            "citation": None,
            "gated": True,
            "s_max": round(s_max, 4),
            "threshold": active_threshold,
            "retrieved_candidates": len(docs),
            "reason": f"Maximum candidate cosine similarity ({s_max:.4f}) is below threshold ({active_threshold}). LLM invocation suppressed."
        }

    # 5. Best candidate page and doc provenance
    best_idx = int(np.argmax(scores))
    best_meta = metadatas[best_idx]
    best_doc_uuid = best_meta.get("doc_uuid", "UNKNOWN_DOC")
    best_source_page = best_meta.get("source_page", 1)
    best_filename = best_meta.get("filename", "")
    target_citation = f"[{best_doc_uuid}, Page {best_source_page}]"

    # 6. Delimiter-fenced bounded prompt construction
    context_chunks_formatted = []
    for i, (doc_text, meta, score) in enumerate(zip(docs, metadatas, scores)):
        c_doc = meta.get("doc_uuid", "UNKNOWN")
        c_page = meta.get("source_page", 1)
        context_chunks_formatted.append(
            f"[Chunk {i+1} - sim={score:.3f}] Source: {c_doc}, Page {c_page}\n{doc_text}"
        )

    fenced_context = "\n\n".join(context_chunks_formatted)

    prompt = (
        f"<CONTEXT>\n"
        f"{fenced_context}\n"
        f"</CONTEXT>\n\n"
        f"Answer the question using only the context above. Be concise and accurate. "
        f"End your answer with:\n"
        f"Citation: {target_citation}\n\n"
        f"Question: {question}"
    )

    # 7. Decoupled local generation via Ollama (llama3)
    llm_error = None
    answer_text = ""
    try:
        raw_response = call_ollama(prompt, model=OLLAMA_MODEL)
        answer_text = raw_response.strip()

        # Enforce citation format if omitted or truncated by LLM
        if f"Citation: {target_citation}" not in answer_text and f"Citation: [{best_doc_uuid}" not in answer_text:
            answer_text = f"{answer_text}\n\n| Citation: {target_citation}"
        elif "| Citation:" not in answer_text and "Citation:" in answer_text:
            # Standardize citation delimiter
            answer_text = answer_text.replace("Citation:", "| Citation:")

    except Exception as exc:
        llm_error = str(exc)
        # Graceful fallback when Ollama daemon is not yet running on host:
        # Synthesize extractive grounded response using the top matching chunk
        primary_snippet = docs[best_idx][:400].strip() + "..."
        answer_text = (
            f"(Ollama local LLM service currently offline: {llm_error})\n\n"
            f"Extracted Grounded Evidence: \"{primary_snippet}\"\n\n"
            f"| Citation: {target_citation}"
        )

    return {
        "answer": answer_text,
        "citation": target_citation,
        "gated": False,
        "s_max": round(s_max, 4),
        "threshold": RELEVANCE_THRESHOLD,
        "best_source": {
            "doc_uuid": best_doc_uuid,
            "filename": best_filename,
            "page": best_source_page,
            "score": round(s_max, 4)
        },
        "retrieved_candidates": len(docs),
        "context_debug": [
            {
                "chunk_idx": meta.get("chunk_idx"),
                "source_page": meta.get("source_page"),
                "score": round(score, 4),
                "preview": text[:140] + "..."
            }
            for text, meta, score in zip(docs, metadatas, scores)
        ],
        "ollama_connected": llm_error is None
    }


@app.get("/api/system-status")
async def system_status(auth_data: Optional[Dict[str, Any]] = Depends(verify_token)):
    """
    Returns system readiness status including ChromaDB count and Ollama connectivity.
    """
    tenant_id = auth_data["tenant_id"] if auth_data else "unauthenticated"
    collection = get_collection()
    total_vectors = collection.count()

    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "status": "ready",
        "chroma_total_vectors": total_vectors,
        "relevance_threshold_tau": RELEVANCE_THRESHOLD,
        "ollama_connected": ollama_ok,
        "active_tenant": tenant_id
    }


# Mount static assets and serve single-page frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Secure Multi-Tenant RAG API running. Frontend index.html not yet initialized."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8008))
    uvicorn.run(app, host="127.0.0.1", port=port)
