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

1. `IngestionAgent`
   Processes uploaded files, extracts text or image descriptions, chunks content, and stores indexed records.
2. `RetrievalAgent`
   Searches indexed chunks and retrieves the most relevant evidence for a question.
3. `ReasoningAgent`
   Produces the final answer from retrieved evidence only, with confidence and clarification signals.
4. `CitationAgent`
   Formats evidence into user-facing references and snippets.

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
- Pinecone provides vector retrieval, and the app can still fall back safely when cloud services are unavailable

## Tech Stack

- FastAPI backend
- LangGraph for multi-agent orchestration
- Pinecone vector database
- Vertex AI Gemini for reasoning and image understanding
- Vertex AI text embeddings
- PyMuPDF for PDF parsing
- Pillow for image metadata handling and safe image fallback
- Postman collection for API testing/demo

## Local Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy [.env.example](/Users/Apple/Documents/ecommerce_agent/.env.example) to `.env` and fill in your credentials:

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

Open [http://localhost:8000](http://localhost:8000). The root endpoint returns API status information, and the main demo flow is through the API/Postman requests below.

## Docker

### Build

```bash
docker build -t ecommerce-support-ai .
```

### Run

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

### `POST /api/upload`

Multipart form upload:

- field name: `file`

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

## Postman

Ready-to-import Postman files are included in [postman/Ecommerce-Support-API.postman_collection.json](/Users/Apple/Documents/ecommerce_agent/postman/Ecommerce-Support-API.postman_collection.json) and [postman/Ecommerce-Support-Local.postman_environment.json](/Users/Apple/Documents/ecommerce_agent/postman/Ecommerce-Support-Local.postman_environment.json).

### Import and run

1. Start the backend:

```bash
uvicorn app.main:app --reload --reload-dir app --reload-dir tests
```

2. Open Postman and import:

- `postman/Ecommerce-Support-API.postman_collection.json`
- `postman/Ecommerce-Support-Local.postman_environment.json`

3. Select the `Ecommerce Support Local` environment.

4. Set `uploadFilePath` to an absolute file path on your machine, for example:

```text
/Users/Apple/Documents/sample_data/invoice.pdf
```

### Included requests

- `Health Check`
- `Upload File`
- `Query Without Upload`
- `Query Using Uploaded File`

`Upload File` stores the returned `file_id` into the collection variable `lastFileId`, and `Query Using Uploaded File` reuses it automatically.

### `POST /api/query`

```json
{
  "question": "I uploaded my invoice. Am I eligible for a refund?",
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

### Implemented evaluation metrics

- `evidence_count`: how many chunks were actually used for the answer
- `retrieved_candidate_count`: how many chunks survived retrieval/reranking for the final response
- `file_scope_applied`: whether retrieval was constrained to explicit `file_ids`
- `top_relevance_score`: strongest retrieval match score
- `average_relevance_score`: average score across returned evidence
- `grounded_response`: whether the answer includes supporting references
- `clarification_rate`: `1.0` when the system asks for clarification, otherwise `0.0`
- `response_latency_ms`: end-to-end response latency for the query call

## Failure Handling

- If no relevant evidence is found, the system says it cannot answer confidently
- If a question is ambiguous, the system requests clarification
- If Vertex AI image analysis fails, the app falls back to safe image metadata summaries instead of crashing
- Answers are generated from retrieved evidence only

## Performance Notes

- Pinecone-backed retrieval keeps query-time search fast once files are indexed
- Chunking prevents very large documents from overwhelming retrieval
- The architecture can later add async ingestion, reranking, and background jobs

## Suggested Demo Video

In 2 minutes, show:

1. App startup
2. Upload an invoice or refund policy file
3. Upload a damaged-product image
4. Ask a support question
5. Show references in the response
6. Ask an ambiguous question to demonstrate fallback behavior

## Trade-Offs

- Pinecone and Vertex AI improve realism, but they increase setup complexity versus a pure local demo
- Fallback image understanding is intentionally conservative when cloud image analysis is unavailable
- The current graph is sequential by design, which keeps it simple and explainable but leaves room for richer branching logic

For a production version, the next upgrades would be:

- PostgreSQL or object storage for file metadata
- Metadata-based retrieval filtering and reranking
- Background ingestion jobs
- OCR pipeline for image text extraction
- Authentication and per-user document isolation
