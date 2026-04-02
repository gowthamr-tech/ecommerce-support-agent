from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

from PIL import Image

from app.services.vertex_ai_service import VertexAIService


class ImageAnalyzer:
    def __init__(self) -> None:
        self.vertex_service = VertexAIService()

    def analyze(self, file_path: Path) -> str:
        analyzed = self._analyze_with_llm(file_path)
        if analyzed:
            return analyzed
        return self._fallback_summary(file_path)

    def _analyze_with_llm(self, file_path: Path) -> Optional[str]:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "image/png"
        return self.vertex_service.analyze_image(
            file_bytes=file_path.read_bytes(),
            mime_type=mime_type,
            filename=file_path.name,
        )

    def _fallback_summary(self, file_path: Path) -> str:
        with Image.open(file_path) as image:
            width, height = image.size
            mode = image.mode
        return (
            f"Image uploaded: {file_path.name}. "
            f"Basic metadata only is available without a vision API key. "
            f"Dimensions: {width}x{height}, color mode: {mode}."
        )
