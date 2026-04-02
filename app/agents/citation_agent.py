from __future__ import annotations

from textwrap import shorten


class CitationAgent:
    def format_references(self, evidence: list[dict]) -> list[dict]:
        references: list[dict] = []
        for item in evidence:
            references.append(
                {
                    "source": item["filename"],
                    "chunk_id": item["chunk_id"],
                    "snippet": shorten(item["content"], width=180, placeholder="..."),
                    "score": item.get("score"),
                }
            )
        return references
