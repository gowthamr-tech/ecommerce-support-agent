import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agents.orchestrator import SupportOrchestrator
from app.config import get_settings
from app.models.schemas import CombinedResponse, UploadResponse

router = APIRouter(tags=["ask"])
orchestrator = SupportOrchestrator()

ALLOWED_SUFFIXES = {".pdf", ".txt", ".csv", ".md", ".png", ".jpg", ".jpeg", ".webp", ".gif"}


@router.post("/ask", response_model=CombinedResponse)
async def ask(
    question: str = Form(..., min_length=3),
    files: List[UploadFile] = File(...),
) -> CombinedResponse:
    """Upload one or more files (PDF, TXT, CSV, MD, images) and ask a question in one request."""
    settings = get_settings()

    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    uploads: list[UploadResponse] = []
    file_ids: list[str] = []

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="One or more files is missing a filename.")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix}' for '{file.filename}'. "
                       f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
            )

        stored_name = f"{uuid.uuid4().hex[:12]}{suffix}"
        destination = settings.uploads_dir / stored_name

        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = orchestrator.ingest_file(file_path=destination, original_name=file.filename)
        uploads.append(UploadResponse(**result))
        file_ids.append(result["file_id"])

    query_result = orchestrator.answer_question(question=question, file_ids=file_ids)

    return CombinedResponse(
        uploads=uploads,
        **query_result,
    )
