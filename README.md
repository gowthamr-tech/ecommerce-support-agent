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
- Image handling is supported through a vision-capable LLM when an API key is available
- The system still works without an API key using deterministic fallback logic

## Tech Stack

- FastAPI backend
- Static HTML/CSS/JS UI
- Local JSON-backed vector store
- PyMuPDF for PDF parsing
- Pillow for image metadata handling
- Optional OpenAI multimodal reasoning for richer answers

## Local Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Optional environment variables

Create a `.env` file if you want richer reasoning and image analysis:

```env
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4.1-mini
```

### 3. Run the app

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

## Docker

### Build

```bash
docker build -t ecommerce-support-ai .
```

### Run

```bash
docker run -p 8000:8000 --env OPENAI_API_KEY=$OPENAI_API_KEY ecommerce-support-ai
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
  "reasoning_summary": "Used 3 evidence chunk(s)..."
}
```

## Failure Handling

- If no relevant evidence is found, the system says it cannot answer confidently
- If a question is ambiguous, the system requests clarification
- If no vision API key is available, the app falls back to safe image metadata summaries
- Answers are generated from retrieved evidence only

## Performance Notes

- Lightweight local search keeps the demo simple and fast
- Chunking prevents very large documents from overwhelming retrieval
- The architecture can later swap in a real vector database and async task queue

## Suggested Demo Video

In 2 minutes, show:

1. App startup
2. Upload an invoice or refund policy file
3. Upload a damaged-product image
4. Ask a support question
5. Show references in the response
6. Ask an ambiguous question to demonstrate fallback behavior

## Trade-Offs

- Local JSON storage is simple for evaluation, not for production scale
- Fallback image understanding is intentionally conservative without a vision model
- The current retrieval is lightweight lexical similarity rather than full semantic embeddings

For a production version, the next upgrades would be:

- PostgreSQL or object storage for file metadata
- A true vector database
- Background ingestion jobs
- OCR pipeline for image text extraction
- Authentication and per-user document isolation
