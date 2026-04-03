from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.services.embedder import cosine_similarity, term_frequency, tokenize
from app.services.storage import JsonStore


class LocalVectorStore:
    def __init__(self, storage_dir: Path):
        self.store = JsonStore(storage_dir / "chunks.json")

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        current = self.store.read()
        current.extend(chunks)
        self.store.write(current)

    def all_chunks(self) -> list[dict[str, Any]]:
        return self.store.read()

    def search(
        self,
        question: str,
        file_ids: Optional[list[str]] = None,
        top_k: int = 5,
        media_type_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        question_vector = term_frequency(tokenize(question))
        matches: list[dict[str, Any]] = []
        for item in self.all_chunks():
            if file_ids and item["file_id"] not in file_ids:
                continue
            if media_type_filter and item.get("media_type") != media_type_filter:
                continue
            score = cosine_similarity(question_vector, term_frequency(tokenize(item["content"])))
            if score <= 0:
                continue
            enriched = dict(item)
            enriched["score"] = round(score, 4)
            matches.append(enriched)
        matches.sort(key=lambda row: row["score"], reverse=True)
        return matches[:top_k]
