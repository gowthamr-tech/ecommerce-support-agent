# E-Commerce Support Multi-Agent System

A production-oriented multi-agent AI support system for e-commerce scenarios. The app accepts document and image uploads, indexes their content, retrieves relevant evidence, and returns grounded answers with references.

## What It Does

- Upload documents such as invoices, receipts, and policy files
- Upload images such as damaged-product photos or screenshots
- Ask support questions in natural language
- Generate grounded responses with supporting references
- Handle ambiguity and missing information with graceful fallback

## Multi-Agent Design

This implementation uses four collaborating agents:

1. `IngestionAgent` — Processes uploaded files, extracts text or image descriptions, chunks content, and stores indexed records.
2. `RetrievalAgent` — Searches indexed chunks and retrieves the most relevant evidence for a question.
3. `ReasoningAgent` — Produces the final answer from retrieved evidence only, with confidence and clarification signals.
4. `CitationAgent` — Formats evidence into user-facing references and snippets.

### Coordination Flow

1. User uploads a file through the UI or API.
2. `IngestionAgent` extracts content and stores chunks.
3. User submits a support question.
4. `RetrievalAgent` finds relevant document and image-derived evidence.
5. `ReasoningAgent` drafts a grounded answer.
6. `CitationAgent` returns supporting references.

## Why This Design

- Modular responsibilities make the system easy to extend
- Retrieval grounds the answer and reduces hallucination
- Image handling is supported through Vertex AI Gemini with safe fallback behavior
- Pinecone provides vector retrieval, and the app falls back safely when cloud services are unavailable

## Tech Stack

- FastAPI backend
- LangGraph for multi-agent orchestration
- Pinecone vector database
- Vertex AI Gemini for reasoning and image understanding
- Vertex AI text embeddings
- PyMuPDF for PDF parsing
- Pillow for image metadata handling and safe image fallback
- Postman collection for API testing/demo

## Project Structure

```
app/
├── main.py                  # FastAPI app entry point
├── config.py                # Settings via Pydantic
├── api/
│   └── routes.py            # All API endpoints in one file
├── agents/
│   ├── orchestrator.py
│   ├── ingestion_agent.py
│   ├── retrieval_agent.py
│   ├── reasoning_agent.py
│   └── citation_agent.py
├── services/
│   ├── vector_store.py      # Local vector store
│   ├── pinecone_store.py    # Pinecone with local fallback
│   ├── vertex_ai_service.py
│   ├── embedder.py
│   ├── image_analyzer.py
│   ├── parser.py
│   └── storage.py
└── models/
    └── schemas.py           # Pydantic request/response schemas
tests/
├── conftest.py
├── test_health.py
└── test_evaluation_metrics.py
```

## Local Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```env
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=rag-ai
PINECONE_NAMESPACE=default
GEMINI_MODEL=gemini-2.5-pro
EMBEDDING_MODEL=text-embedding-004
```

### 3. Run the app

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

## Docker

```bash
docker build -t ecommerce-support-ai .
```

```bash
docker run -p 8000:8000 \
  --env GCP_PROJECT_ID=$GCP_PROJECT_ID \
  --env GCP_LOCATION=$GCP_LOCATION \
  --env PINECONE_API_KEY=$PINECONE_API_KEY \
  --env PINECONE_INDEX_NAME=$PINECONE_INDEX_NAME \
  --env PINECONE_NAMESPACE=$PINECONE_NAMESPACE \
  --env GEMINI_MODEL=$GEMINI_MODEL \
  --env EMBEDDING_MODEL=$EMBEDDING_MODEL \
  ecommerce-support-ai
```

Or:

```bash
docker compose up --build
```

## API

### `GET /health`

```json
{ "status": "ok" }
```

---

### `POST /api/upload`

Multipart form upload. Field name: `file`.

Supported types: `.pdf`, `.txt`, `.csv`, `.md`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`

Response:

```json
{
  "file_id": "abc123def456",
  "filename": "invoice.pdf",
  "media_type": "document",
  "extracted_summary": "Purchase date: March 21, 2026 ...",
  "chunks_indexed": 3
}
```

---

### `POST /api/query`

Request:

```json
{
  "question": "Am I eligible for a refund?",
  "file_ids": ["abc123def456"]
}
```

Response:

```json
{
  "answer": "Based on the uploaded material and matching policy content...",
  "confidence": "medium",
  "needs_clarification": false,
  "references": [
    {
      "source": "refund_policy.txt",
      "chunk_id": "xyz-0",
      "snippet": "Customers may request a refund within 7 calendar days...",
      "score": 0.31
    }
  ],
  "reasoning_summary": "Used 3 evidence chunk(s)...",
  "runtime": {
    "vector_backend": "pinecone",
    "llm_backend": "vertex-ai-gemini",
    "orchestration_backend": "langgraph"
  },
  "evaluation": {
    "evidence_count": 3,
    "retrieved_candidate_count": 3,
    "file_scope_applied": true,
    "top_relevance_score": 0.31,
    "average_relevance_score": 0.24,
    "grounded_response": true,
    "clarification_rate": 0.0,
    "response_latency_ms": 182.4
  }
}
```

---

### `POST /api/ask`

Upload one or more files and ask a question in a single request. Multipart form fields:

- `question` (string, min 3 chars)
- `files` (one or more files)

Response combines upload results and query answer:

```json
{
  "uploads": [
    {
      "file_id": "abc123def456",
      "filename": "invoice.pdf",
      "media_type": "document",
      "extracted_summary": "Purchase date: March 21, 2026 ...",
      "chunks_indexed": 3
    }
  ],
  "answer": "Based on the uploaded invoice...",
  "confidence": "high",
  "needs_clarification": false,
  "references": [...],
  "reasoning_summary": "Used 2 evidence chunk(s)...",
  "runtime": {...},
  "evaluation": {...}
}
```

---

### Evaluation Metrics

| Field | Description |
|---|---|
| `evidence_count` | Number of chunks used for the answer |
| `retrieved_candidate_count` | Chunks surviving retrieval/reranking |
| `file_scope_applied` | Whether retrieval was scoped to explicit `file_ids` |
| `top_relevance_score` | Strongest retrieval match score |
| `average_relevance_score` | Average score across returned evidence |
| `grounded_response` | Whether the answer includes supporting references |
| `clarification_rate` | `1.0` when clarification is requested, otherwise `0.0` |
| `response_latency_ms` | End-to-end response latency |

## Postman

Ready-to-import files are in the `postman/` directory:

- `postman/Ecommerce-Support-API.postman_collection.json`
- `postman/Ecommerce-Support-Local.postman_environment.json`

### Import and run

1. Start the backend:

```bash
uvicorn app.main:app --reload
```

2. Open Postman and import both files above.
3. Select the `Ecommerce Support Local` environment.
4. Set `uploadFilePath` to an absolute path on your machine, e.g. `/Users/you/sample_data/invoice.pdf`.

### Included requests

- `Health Check`
- `Upload File`
- `Query Without Upload`
- `Query Using Uploaded File`

`Upload File` stores the returned `file_id` into `lastFileId`, and `Query Using Uploaded File` reuses it automatically.

## Tests

```bash
pytest tests/
```

- `test_health.py` — boots the full FastAPI app and hits `/health`
- `test_evaluation_metrics.py` — unit tests for metric calculations and file scope resolution

## Failure Handling

- No relevant evidence → system says it cannot answer confidently
- Ambiguous question → system requests clarification
- Vertex AI image analysis fails → falls back to safe image metadata summary
- Pinecone unavailable → falls back to local vector store
- Answers are generated from retrieved evidence only

## Trade-Offs

- Pinecone and Vertex AI improve realism but increase setup complexity versus a pure local demo
- Fallback image understanding is intentionally conservative when cloud image analysis is unavailable
- The current graph is sequential by design, which keeps it simple and explainable but leaves room for richer branching logic

## Suggested Next Steps

- PostgreSQL or object storage for file metadata
- Background ingestion jobs
- OCR pipeline for image text extraction
- Metadata-based retrieval filtering and reranking
- Authentication and per-user document isolation
