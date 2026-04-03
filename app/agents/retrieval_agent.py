from __future__ import annotations

from typing import Any, Optional

from app.config import get_settings
from app.services.pinecone_store import PineconeVectorStore


POLICY_KEYWORDS = {"refund", "rejected", "warranty", "eligible", "covered", "policy", "invoice"}
IMAGE_KEYWORDS = {"image", "photo", "picture", "screenshot", "damaged", "damage", "error"}


class RetrievalAgent:
    def __init__(self) -> None:
        settings = get_settings()
        self.top_k = settings.retrieval_top_k
        self.vector_store = PineconeVectorStore()

    def retrieve(self, question: str, file_ids: Optional[list[str]] = None) -> list[dict]:
        raw_matches = self.vector_store.search(question=question, file_ids=file_ids, top_k=max(self.top_k * 3, 15))

        # When the question is about an image and files are scoped, ensure image chunks
        # are always included — Pinecone embedding similarity can miss them otherwise.
        if file_ids and any(kw in question.lower() for kw in IMAGE_KEYWORDS):
            seen_ids = {m.get("chunk_id") for m in raw_matches}
            image_chunks = self.vector_store.search(
                question=question, file_ids=file_ids, top_k=max(self.top_k * 3, 15),
                media_type_filter="image",
            )
            for chunk in image_chunks:
                if chunk.get("chunk_id") not in seen_ids:
                    raw_matches.append(chunk)

        ranked = self._rerank(question=question, matches=raw_matches)
        return ranked[: self.top_k]

    def backend_name(self) -> str:
        return self.vector_store.backend_name()

    def _rerank(self, question: str, matches: list[dict[str, Any]]) -> list[dict]:
        lowered = question.lower()
        asks_about_images = any(keyword in lowered for keyword in IMAGE_KEYWORDS)
        asks_about_policy = any(keyword in lowered for keyword in POLICY_KEYWORDS)

        reranked: list[dict[str, Any]] = []
        for item in matches:
            adjusted = dict(item)
            score = float(item.get("score", 0.0))
            media_type = item.get("media_type", "")
            filename = str(item.get("filename", "")).lower()

            if asks_about_policy and not asks_about_images:
                if media_type in {"document", "text"}:
                    score += 0.2
                if media_type == "image":
                    score -= 0.15
                if any(token in filename for token in ("policy", "faq", "refund", "warranty")):
                    score += 0.25

            if asks_about_images and media_type == "image":
                score += 0.2

            adjusted["score"] = round(score, 4)
            reranked.append(adjusted)

        reranked.sort(key=lambda row: row.get("score", 0.0), reverse=True)

        if asks_about_policy and not asks_about_images:
            reranked = [
                item
                for item in reranked
                if not (item.get("media_type") == "image" and float(item.get("score", 0.0)) < 0.45)
            ]

        return reranked
