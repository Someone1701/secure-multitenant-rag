import io
import re
import uuid
import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

import pymupdf as fitz  # PyMuPDF
import docx  # python-docx
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB persistent storage in ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")
CHROMA_COLLECTION_NAME = "rag_chunks"

# Lazy-loaded Sentence-BERT bi-encoder model (all-MiniLM-L6-v2)
_embed_model = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def get_collection():
    """Returns or creates the unified multi-tenant collection."""
    return chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def sanitize_text(raw_text: str) -> str:
    """
    Prompt injection defense:
    Strips or neutralizes adversarial patterns such as </CONTEXT>, <SYSTEM>,
    IGNORE PREVIOUS INSTRUCTIONS, etc., before text enters chunking or embedding.
    """
    if not raw_text:
        return ""

    # Neutralize XML/tag-like prompt boundaries
    cleaned = re.sub(r'</?(?:context|system|instruction|prompt|assistant|human)[^>]*>', ' ', raw_text, flags=re.IGNORECASE)

    # Neutralize common indirect prompt injection attack phrases
    injection_patterns = [
        r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?',
        r'disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?',
        r'you\s+are\s+now\s+in\s+developer\s+mode',
        r'system\s*:\s*override',
    ]
    for pat in injection_patterns:
        cleaned = re.sub(pat, '[FILTERED_INSTRUCTION]', cleaned, flags=re.IGNORECASE)

    # Collapse multiple whitespaces
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    return cleaned.strip()


def parse_pdf_layout_preserving(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    PDF Parsing with PyMuPDF (fitz):
    Examines glyph bounding boxes (x0, y0, x1, y1) to detect column gutters
    and reconstruct reading order column-by-column rather than raw stream order.
    Tracks physical 1-indexed source_page.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_data: List[Dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        # Extract structured text blocks: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        # Filter for text blocks (block_type == 0)
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        if not text_blocks:
            continue

        # Sort blocks to preserve column gutters:
        # Determine average page width to group columns if multi-column
        rect = page.rect
        mid_x = rect.width / 2.0

        # Heuristic for multi-column detection: check if blocks are distinctly separated horizontally
        left_column = []
        right_column = []
        full_width = []

        is_multi_column = False
        left_count = sum(1 for b in text_blocks if b[2] <= mid_x + 20)
        right_count = sum(1 for b in text_blocks if b[0] >= mid_x - 20)

        # If significant blocks exist in distinct left and right halves
        if left_count > 2 and right_count > 2 and (left_count + right_count) >= len(text_blocks) * 0.7:
            is_multi_column = True

        if is_multi_column:
            for b in text_blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                if x1 <= mid_x + 10:
                    left_column.append((y0, x0, text))
                elif x0 >= mid_x - 10:
                    right_column.append((y0, x0, text))
                else:
                    full_width.append((y0, x0, text))

            # Sort by top-to-bottom within columns
            left_column.sort(key=lambda item: item[0])
            right_column.sort(key=lambda item: item[0])
            full_width.sort(key=lambda item: item[0])

            # In natural order: headers/full-width at top, left column, right column, footer
            ordered_blocks = [item[2] for item in full_width if item[0] < rect.height * 0.25]
            ordered_blocks.extend([item[2] for item in left_column])
            ordered_blocks.extend([item[2] for item in right_column])
            ordered_blocks.extend([item[2] for item in full_width if item[0] >= rect.height * 0.25])
        else:
            # Single column standard top-to-bottom reading order
            sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))
            ordered_blocks = [b[4] for b in sorted_blocks]

        sanitized_page_text = sanitize_text("\n".join(ordered_blocks))
        if sanitized_page_text:
            pages_data.append({
                "page_num": page_num,
                "text": sanitized_page_text
            })

    doc.close()
    return pages_data


def parse_docx_dom(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    DOCX Parsing with python-docx:
    Traverses OpenXML DOM tree elements (paragraphs, tables, headings)
    and estimates page boundaries from form/page breaks.
    """
    doc_file = io.BytesIO(file_bytes)
    doc = docx.Document(doc_file)

    pages_data: List[Dict[str, Any]] = []
    current_page = 1
    current_page_text: List[str] = []

    for element in doc.element.body:
        # Check if element is a paragraph
        if element.tag.endswith('p'):
            # Detect explicit page break XML tags: <w:br w:type="page"/>
            xml_str = element.xml
            has_page_break = 'w:type="page"' in xml_str or 'w:lastRenderedPageBreak' in xml_str

            para_text = element.text.strip()
            if para_text:
                current_page_text.append(para_text)

            if has_page_break and current_page_text:
                page_content = sanitize_text("\n".join(current_page_text))
                if page_content:
                    pages_data.append({"page_num": current_page, "text": page_content})
                current_page += 1
                current_page_text = []

        # Check if element is a table
        elif element.tag.endswith('tbl'):
            table_lines = []
            for row in element.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr'):
                cells = [cell.text.strip() for cell in row.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc')]
                if any(cells):
                    table_lines.append(" | ".join(cells))
            if table_lines:
                current_page_text.append("\n".join(table_lines))

    if current_page_text:
        page_content = sanitize_text("\n".join(current_page_text))
        if page_content:
            pages_data.append({"page_num": current_page, "text": page_content})

    if not pages_data and current_page_text:
        pages_data.append({"page_num": 1, "text": sanitize_text("\n".join(current_page_text))})

    return pages_data


def tokenize_words(text: str) -> List[str]:
    """Tokenizes text by whitespace preserving word tokens."""
    return text.split()


def sliding_window_chunk(
    pages_data: List[Dict[str, Any]],
    window_capacity: int = 512,
    overlap_tokens: int = 64
) -> List[Dict[str, Any]]:
    """
    Sliding window chunker:
    S = 512 tokens capacity, O = 64 tokens overlap, step = S - O = 448 tokens.
    Calculates Chunk_i = [i * 448, i * 448 + 512].
    Preserves exact source_page provenance by mapping token offsets back to page indices.
    """
    step = window_capacity - overlap_tokens  # 448 tokens

    # Build continuous token stream with page tag for each token
    all_tokens: List[Tuple[str, int]] = []
    for page_item in pages_data:
        p_num = page_item["page_num"]
        words = tokenize_words(page_item["text"])
        for w in words:
            all_tokens.append((w, p_num))

    if not all_tokens:
        return []

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0
    total_tokens = len(all_tokens)

    for start in range(0, total_tokens, step):
        end = min(start + window_capacity, total_tokens)
        token_slice = all_tokens[start:end]

        chunk_words = [t[0] for t in token_slice]
        chunk_text = " ".join(chunk_words)

        # Determine dominant source page for this chunk
        pages_in_chunk = [t[1] for t in token_slice]
        # Most frequent page in the window
        dominant_page = max(set(pages_in_chunk), key=pages_in_chunk.count) if pages_in_chunk else 1

        chunks.append({
            "chunk_idx": chunk_idx,
            "text": chunk_text,
            "source_page": dominant_page,
            "token_count": len(chunk_words)
        })
        chunk_idx += 1

        if end >= total_tokens:
            break

    return chunks


def compute_normalized_embeddings(texts: List[str]) -> np.ndarray:
    """
    Sentence-BERT bi-encoder (all-MiniLM-L6-v2) embedding with L2 normalization (||v||_2 = 1).
    """
    model = get_embed_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # L2 normalize each vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = embeddings / norms
    return normalized_embeddings


def ingest_document(
    file_bytes: bytes,
    filename: str,
    tenant_id: str,
    doc_uuid: Optional[str] = None
) -> Dict[str, Any]:
    """
    Full Ingestion Pipeline:
    1. Parse PDF (glyph-order) or DOCX (DOM tree)
    2. Prompt injection sanitization
    3. Sliding-window chunking (S=512, O=64, step=448)
    4. Sentence-BERT bi-encoder embedding with L2 normalization
    5. Write to ChromaDB collection with persistent metadata:
       {tenant_id, doc_uuid, filename, chunk_idx, source_page, timestamp}
    """
    if not doc_uuid:
        doc_uuid = str(uuid.uuid4())

    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        pages_data = parse_pdf_layout_preserving(file_bytes)
    elif lower_name.endswith(".docx"):
        pages_data = parse_docx_dom(file_bytes)
    else:
        # Fallback text parsing
        raw_text = file_bytes.decode("utf-8", errors="ignore")
        pages_data = [{"page_num": 1, "text": sanitize_text(raw_text)}]

    if not pages_data:
        raise ValueError("Could not extract any readable text from the document.")

    # Sliding-window chunking
    chunks = sliding_window_chunk(pages_data, window_capacity=512, overlap_tokens=64)
    if not chunks:
        raise ValueError("Document yielded no chunks after processing.")

    # Embedding with L2 normalization
    chunk_texts = [c["text"] for c in chunks]
    embeddings = compute_normalized_embeddings(chunk_texts)

    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    collection = get_collection()

    ids = [f"{tenant_id}_{doc_uuid}_{c['chunk_idx']}" for c in chunks]
    metadatas = [
        {
            "tenant_id": tenant_id,
            "doc_uuid": doc_uuid,
            "filename": filename,
            "chunk_idx": int(c["chunk_idx"]),
            "source_page": int(c["source_page"]),
            "timestamp": timestamp_iso
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunk_texts,
        metadatas=metadatas
    )

    return {
        "doc_uuid": doc_uuid,
        "filename": filename,
        "total_pages": len(pages_data),
        "total_chunks": len(chunks),
        "tenant_id": tenant_id,
        "timestamp": timestamp_iso
    }


def list_tenant_documents(tenant_id: str) -> List[Dict[str, Any]]:
    """
    Lists unique documents belonging strictly to the specified tenant.
    Enforces storage-level metadata filtering.
    """
    collection = get_collection()
    # Query all vectors for this tenant
    res = collection.get(
        where={"tenant_id": tenant_id},
        include=["metadatas"]
    )

    doc_map: Dict[str, Dict[str, Any]] = {}
    if res and res.get("metadatas"):
        for meta in res["metadatas"]:
            doc_id = meta.get("doc_uuid")
            if not doc_id:
                continue
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_uuid": doc_id,
                    "filename": meta.get("filename", "Unknown Document"),
                    "chunk_count": 0,
                    "max_page": 1,
                    "timestamp": meta.get("timestamp", "")
                }
            doc_map[doc_id]["chunk_count"] += 1
            source_page = meta.get("source_page", 1)
            if source_page > doc_map[doc_id]["max_page"]:
                doc_map[doc_id]["max_page"] = source_page

    return list(doc_map.values())


def delete_tenant_document(tenant_id: str, doc_uuid: str) -> int:
    """
    Deletes all vector chunks belonging to the given doc_uuid strictly under tenant_id.
    """
    collection = get_collection()
    # Get IDs first to return deleted count
    res = collection.get(
        where={
            "$and": [
                {"tenant_id": tenant_id},
                {"doc_uuid": doc_uuid}
            ]
        }
    )
    ids_to_delete = res.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)
