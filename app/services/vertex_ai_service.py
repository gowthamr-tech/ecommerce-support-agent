from __future__ import annotations

import json
import logging
from typing import List, Optional

from app.config import get_settings
from app.services.embedder import term_frequency, tokenize

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
except ImportError:  # pragma: no cover
    vertexai = None
    GenerativeModel = None
    Part = None
    TextEmbeddingInput = None
    TextEmbeddingModel = None


class VertexAIService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        self._model = None
        self._embedding_model = None
        self._available = False
        self._init_clients()

    @property
    def available(self) -> bool:
        return self._available

    def _init_clients(self) -> None:
        if (
            vertexai is None
            or GenerativeModel is None
            or TextEmbeddingModel is None
            or not self.settings.google_cloud_project
        ):
            self._available = False
            return

        if not self._initialized:
            vertexai.init(
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_location,
            )
            self._model = GenerativeModel(self.settings.gemini_model)
            self._embedding_model = TextEmbeddingModel.from_pretrained(self.settings.embedding_model)
            self._initialized = True
            self._available = True

    def embed_text(self, text: str) -> List[float]:
        if self._available and self._embedding_model is not None and TextEmbeddingInput is not None:
            embeddings = self._embedding_model.get_embeddings([TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT")])
            return list(embeddings[0].values)

        fallback = term_frequency(tokenize(text))
        ordered_tokens = sorted(fallback.keys())[:128]
        return [fallback[token] for token in ordered_tokens]

    def embed_query(self, text: str) -> List[float]:
        if self._available and self._embedding_model is not None and TextEmbeddingInput is not None:
            embeddings = self._embedding_model.get_embeddings([TextEmbeddingInput(text, "RETRIEVAL_QUERY")])
            return list(embeddings[0].values)
        return self.embed_text(text)

    def generate_grounded_response(self, question: str, evidence: list[dict]) -> Optional[dict]:
        if not self._available or self._model is None:
            return None

        evidence_block = "\n\n".join(
            [
                "Source: {filename} | Chunk: {chunk_id} | Score: {score}\n{content}".format(
                    filename=item["filename"],
                    chunk_id=item["chunk_id"],
                    score=item.get("score"),
                    content=item["content"],
                )
                for item in evidence
            ]
        )

        prompt = (
            "You are a grounded e-commerce support assistant. "
            "Answer only from the provided evidence. "
            "If the evidence is incomplete, say what is missing. "
            "Return strict JSON with keys: answer, confidence, needs_clarification, reasoning_summary.\n\n"
            f"Question:\n{question}\n\nEvidence:\n{evidence_block}"
        )
        try:
            response = self._model.generate_content(prompt)
            raw = getattr(response, "text", "") or ""
        except Exception as exc:
            self.logger.warning("Vertex AI grounded generation failed: %s", exc)
            return None
        try:
            parsed = json.loads(raw)
            return {
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "needs_clarification": parsed["needs_clarification"],
                "reasoning_summary": parsed["reasoning_summary"],
            }
        except Exception:
            return None

    def analyze_image(self, file_bytes: bytes, mime_type: str, filename: str) -> Optional[str]:
        if not self._available or self._model is None or Part is None:
            return None

        try:
            part = Part.from_data(data=file_bytes, mime_type=mime_type)
            response = self._model.generate_content(
                [
                    "Describe this customer support image, including visible damage, text, or error details. Keep it factual and concise.",
                    part,
                ]
            )
            text = getattr(response, "text", "") or ""
            return text.strip() or None
        except Exception as exc:
            self.logger.warning("Vertex AI image analysis failed for %s: %s", filename, exc)
            return None
