from __future__ import annotations

from textwrap import shorten
from typing import Optional

from app.services.vertex_ai_service import VertexAIService


class ReasoningAgent:
    def __init__(self) -> None:
        self.vertex_service = VertexAIService()

    def answer(self, question: str, evidence: list[dict]) -> dict:
        if not evidence:
            return {
                "answer": (
                    "I couldn’t find enough support information in the uploaded files or policies to answer that confidently."
                ),
                "confidence": "low",
                "needs_clarification": True,
                "reasoning_summary": "No relevant evidence was retrieved.",
            }

        if self.vertex_service.available:
            llm_result = self._answer_with_llm(question, evidence)
            if llm_result:
                return llm_result
        return self._answer_with_rules(question, evidence)

    def backend_name(self) -> str:
        return "vertex-ai-gemini" if self.vertex_service.available else "rule-based-fallback"

    def _answer_with_llm(self, question: str, evidence: list[dict]) -> Optional[dict]:
        return self.vertex_service.generate_grounded_response(question=question, evidence=evidence)

    def _answer_with_rules(self, question: str, evidence: list[dict]) -> dict:
        top = evidence[0]
        snippets = [shorten(item["content"], width=180, placeholder="...") for item in evidence[:3]]
        ambiguity_markers = ["eligible", "covered", "rejected", "why", "what should i do"]
        needs_clarification = any(marker in question.lower() for marker in ambiguity_markers) and top["score"] < 0.18
        confidence = "high" if top["score"] > 0.3 else "medium" if top["score"] > 0.15 else "low"

        if needs_clarification:
            answer = (
                "I found some related support information, but I still need more detail to answer reliably. "
                "Please share the order, product, or policy detail you want me to check."
            )
        else:
            answer = (
                "Based on the uploaded material and matching policy content, the strongest evidence suggests: "
                f"{snippets[0]}"
            )
            if len(snippets) > 1:
                answer += f" Additional supporting context: {snippets[1]}"

        reasoning_summary = (
            f"Used {len(evidence)} evidence chunk(s). Top evidence came from {top['filename']} with relevance score {top['score']}."
        )
        return {
            "answer": answer,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "reasoning_summary": reasoning_summary,
        }
