# Secure Multi-Tenant Retrieval-Augmented Generation (RAG) System

A working full-stack prototype of a secure, multi-tenant document question-answering architecture implementing **storage-level pre-traversal tenant isolation**, a **deterministic relevance gate ($\tau = 0.65$)**, **layout-preserving multi-format ingestion**, and **page-coordinate provenance tracking**.

---

## 🌟 Key Architectural Features

1. **Storage-Level Pre-Traversal Tenant Isolation**
   - Tenant identities are extracted strictly from cryptographically signed JWTs (never from unverified request parameters).
   - In ChromaDB's HNSW vector index, tenant predicates (`where={"tenant_id": token_tenant_id}`) are evaluated *before* candidate distance computation, preventing unauthorized vector exploration across tenants.

2. **Deterministic Relevance Gate ($\tau = 0.65$)**
   - Evaluates the maximum cosine similarity score ($s_{\max}$) among top candidate vectors.
   - If $s_{\max} < 0.65$, execution is deterministically terminated, returning `"Information not found in source material."` without calling the LLM. This suppresses hallucinations when evidence is absent or insufficient.

3. **Layout-Preserving Ingestion & Page Provenance**
   - **PDFs:** PyMuPDF (`fitz`) evaluates glyph bounding boxes $(x_0, y_0, x_1, y_1)$ to detect column gutters and reconstruct multi-column reading flow.
   - **DOCX:** Traverses the OpenXML DOM tree for paragraphs, headings, and tables.
   - **Chunking:** Sliding window of capacity $S = 512$ tokens and overlap $O = 64$ tokens (stride = 448 tokens), preserving source page coordinates.
   - **Attribution:** Every answer provides an inline page-level citation: `| Citation: [doc_uuid, Page X]`.

4. **Prompt Injection Hardening & Decoupled Inference**
   - Extracted text is stripped of adversarial instructions and structural delimiter injection patterns (`</CONTEXT>`, `<SYSTEM>`, `IGNORE ALL`).
   - Retrieved chunks are encapsulated in strict `<CONTEXT>...</CONTEXT>` delimiter fences.
   - Decoupled inference engine connects to local quantized **LLaMA-3** (via Ollama) for privacy preservation.

---

## 📁 Project Structure

```
CAD/
├── main.py              # FastAPI app, endpoints, similarity gate, Ollama orchestration
├── ingestion.py         # PDF/DOCX layout parsers, chunker, S-BERT embedder, ChromaDB writer
├── auth.py              # JWT signing, claims verification, tenant binding
├── requirements.txt     # Python dependencies
├── README.md            # System documentation and instructions
└── static/
    └── index.html       # Single-page responsive UI with gate visualizer and tenant switcher
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Start Ollama for Local LLaMA-3 Generation
If you want local LLM synthesis, make sure [Ollama](https://ollama.ai) is running and pull the `llama3` model:
```bash
ollama run llama3
```
> *Note: If Ollama is not running, the system will still execute the full vector search, similarity gate, and candidate extraction, gracefully providing the exact grounded chunk and citation.*

### 3. Run the FastAPI Server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the Web Application
Open your browser and navigate to:
```
http://localhost:8000
```

---

## 🧪 Demonstration & Testing Guide

### Scenario 1: Multi-Tenant Isolation
1. Click **Alice** in the top bar to authenticate as `tenant_alice`.
2. Upload a document (e.g., `Cloud Architecture Design Submission 1.pdf`).
3. Click **Bob** in the top bar to switch to `tenant_bob`. Notice that Bob's document list is completely empty.
4. Try asking a question about Alice's document while logged in as Bob. The system will report 0 candidate vectors and refuse the answer.

### Scenario 2: Relevance Grounding Gate ($\tau = 0.65$)
1. Ask a question directly covered by your uploaded document (e.g., *"What is the deterministic similarity gate threshold?"*).
   - Gate status: **GATE PASSED ($s_{\max} \ge 0.65$)**.
   - Model outputs the answer with citation: `| Citation: [doc_uuid, Page X]`.
2. Ask an irrelevant question (e.g., *"What is the stock price of Apple in 2030?"*).
   - Gate status: **REFUSED ($s_{\max} < 0.65$)**.
   - Output: `"Information not found in source material."` with no LLM invocation.

---

## 🔒 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/login` | Issues a signed JWT with `tenant_id` claim |
| `POST` | `/documents/upload` | Ingests, chunks, embeds, and stores PDF/DOCX under tenant |
| `GET` | `/documents` | Lists all documents belonging strictly to authenticated tenant |
| `DELETE` | `/documents/{doc_uuid}` | Deletes document vectors belonging to tenant |
| `POST` | `/query` | Executes pre-traversal search, applies gate $\tau=0.65$, and returns answer |
| `GET` | `/api/system-status` | System health check and vector counts |
