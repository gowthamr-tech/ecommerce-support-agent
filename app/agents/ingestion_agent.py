from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import Literal, Optional

from app.config import get_settings
from app.services.image_analyzer import ImageAnalyzer
from app.services.parser import (
    chunk_text,
    extract_text_from_pdf,
    extract_text_from_plaintext,
    summarize_text,
)
from app.services.storage import JsonStore
from app.services.pinecone_store import PineconeVectorStore


class IngestionAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = PineconeVectorStore()
        self.file_store = JsonStore(self.settings.vectorstore_dir / "files.json")
        self.image_analyzer = ImageAnalyzer()

    def ingest(self, file_path: Path, original_name: Optional[str] = None) -> dict:
        media_type = self._infer_media_type(file_path)
        source_name = original_name or file_path.name
        file_id = uuid.uuid4().hex[:12]

        if media_type == "document":
            extracted_text = extract_text_from_pdf(file_path)
        elif media_type == "text":
            extracted_text = extract_text_from_plaintext(file_path)
        else:
            extracted_text = self.image_analyzer.analyze(file_path)

        chunks = chunk_text(
            extracted_text,
            chunk_size=self.settings.max_chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunk_payload = [
            {
                "chunk_id": f"{file_id}-{index}",
                "file_id": file_id,
                "filename": source_name,
                "media_type": media_type,
                "content": chunk,
            }
            for index, chunk in enumerate(chunks)
        ]
        if chunk_payload:
            self.vector_store.add_chunks(chunk_payload)

        files = self.file_store.read()
        files.append(
            {
                "file_id": file_id,
                "filename": source_name,
                "media_type": media_type,
                "summary": summarize_text(extracted_text),
            }
        )
        self.file_store.write(files)

        return {
            "file_id": file_id,
            "filename": source_name,
            "media_type": media_type,
            "extracted_summary": summarize_text(extracted_text),
            "chunks_indexed": len(chunk_payload),
        }

    def preload_policies(self) -> int:
        self.vector_store.sync_local_chunks()
        existing_files = {row["filename"] for row in self.file_store.read()}
        ingested_count = 0
        for file_path in sorted(self.settings.policies_dir.iterdir()):
            if not file_path.is_file() or file_path.name in existing_files:
                continue
            self.ingest(file_path, original_name=file_path.name)
            ingested_count += 1
        return ingested_count

    def _infer_media_type(self, file_path: Path) -> Literal["document", "image", "text"]:
        mime_type, _ = mimetypes.guess_type(file_path.name)
        if mime_type and mime_type.startswith("image/"):
            return "image"
        if file_path.suffix.lower() == ".pdf":
            return "document"
        return "text"
