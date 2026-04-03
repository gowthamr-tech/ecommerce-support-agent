import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agents.orchestrator import SupportOrchestrator
from app.config import get_settings
from app.models.schemas import UploadResponse

router = APIRouter(tags=["upload"])
orchestrator = SupportOrchestrator()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix or ".bin"
    stored_name = f"{uuid.uuid4().hex[:12]}{suffix}"
    destination = settings.uploads_dir / stored_name
    print("What happen here ", destination)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = orchestrator.ingest_file(file_path=destination, original_name=file.filename)
    return UploadResponse(**result)
