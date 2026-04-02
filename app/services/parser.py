from __future__ import annotations

from pathlib import Path


def extract_text_from_pdf(file_path: Path) -> str:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PDF support requires PyMuPDF. Install dependencies from requirements.txt before uploading PDFs."
        ) from exc

    text_parts: list[str] = []
    with fitz.open(file_path) as document:
        for page in document:
            text_parts.append(page.get_text("text"))
    return "\n".join(part.strip() for part in text_parts if part.strip())


def extract_text_from_plaintext(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(normalized):
        end = start + chunk_size
        chunks.append(normalized[start:end].strip())
        start += step
    return [chunk for chunk in chunks if chunk]


def summarize_text(text: str, max_chars: int = 280) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "No extractable text was found."
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."
