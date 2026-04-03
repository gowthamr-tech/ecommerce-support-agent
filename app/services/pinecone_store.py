from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import get_settings
from app.services.storage import JsonStore
from app.services.vector_store import LocalVectorStore
from app.services.vertex_ai_service import VertexAIService

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:  # pragma: no cover
    Pinecone = None
    ServerlessSpec = None


class PineconeVectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.vertex_service = VertexAIService()
        self.local_store = LocalVectorStore(self.settings.vectorstore_dir)
        self.file_store = JsonStore(self.settings.vectorstore_dir / "files.json")
        self.index = None
        self.available = False
        self.expected_dimension = len(self.vertex_service.embed_text("dimension check"))
        self._init_index()

    def _init_index(self) -> None:
        if Pinecone is None or not self.settings.pinecone_api_key:
            self.available = False
            self.logger.info("Pinecone disabled: missing package or API key")
            return

        try:
            client = Pinecone(api_key=self.settings.pinecone_api_key)
            indexes = client.list_indexes()
            existing_indexes = {item["name"] for item in indexes}
            index_map = {item["name"]: item for item in indexes}
            if self.settings.pinecone_index_name not in existing_indexes and ServerlessSpec is not None:
                client.create_index(
                    name=self.settings.pinecone_index_name,
                    dimension=self.expected_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            elif self.settings.pinecone_index_name in index_map:
                actual_dimension = index_map[self.settings.pinecone_index_name].get("dimension")
                if actual_dimension != self.expected_dimension:
                    self.available = False
                    self.logger.warning(
                        "Pinecone index dimension mismatch for '%s': index=%s embedding=%s. Falling back locally.",
                        self.settings.pinecone_index_name,
                        actual_dimension,
                        self.expected_dimension,
                    )
                    return

            self.index = client.Index(self.settings.pinecone_index_name)
            self.available = True
            self.logger.info("Pinecone enabled for index '%s'", self.settings.pinecone_index_name)
        except Exception as exc:
            self.available = False
            self.index = None
            self.logger.warning("Pinecone init failed: %r", exc)

    def backend_name(self) -> str:
        return "pinecone" if self.available and self.index is not None else "local-fallback"

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self.local_store.add_chunks(chunks)

        if not self.available or self.index is None:
            return

        self._upsert_chunks(chunks)

    def sync_local_chunks(self) -> int:
        chunks = self.local_store.all_chunks()
        if not chunks or not self.available or self.index is None:
            return 0
        self._upsert_chunks(chunks)
        return len(chunks)

    def search(
        self,
        question: str,
        file_ids: Optional[list[str]] = None,
        top_k: int = 5,
        media_type_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not self.available or self.index is None:
            return self.local_store.search(
                question=question, file_ids=file_ids, top_k=top_k, media_type_filter=media_type_filter
            )
        query_vector = self.vertex_service.embed_query(question)
        pinecone_filter: dict = {}
        if file_ids:
            pinecone_filter["file_id"] = {"$in": file_ids}
        if media_type_filter:
            pinecone_filter["media_type"] = {"$eq": media_type_filter}
        result = self.index.query(
            namespace=self.settings.pinecone_namespace,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter if pinecone_filter else None,
        )

        matches = []
        for match in result.get("matches", []):
            metadata = match.get("metadata", {})
            matches.append(
                {
                    "chunk_id": metadata.get("chunk_id", match.get("id")),
                    "file_id": metadata.get("file_id"),
                    "filename": metadata.get("filename"),
                    "media_type": metadata.get("media_type"),
                    "content": metadata.get("content", ""),
                    "score": round(match.get("score", 0.0), 4),
                }
            )
        return matches

    def _upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not self.available or self.index is None:
            return

        vectors = []
        for chunk in chunks:
            embedding = self.vertex_service.embed_text(chunk["content"])
            vectors.append(
                {
                    "id": chunk["chunk_id"],
                    "values": embedding,
                    "metadata": {
                        "file_id": chunk["file_id"],
                        "filename": chunk["filename"],
                        "media_type": chunk["media_type"],
                        "content": chunk["content"],
                        "chunk_id": chunk["chunk_id"],
                    },
                }
            )
        if vectors:
            self.index.upsert(vectors=vectors, namespace=self.settings.pinecone_namespace)
