from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional, TypedDict

from app.agents.citation_agent import CitationAgent
from app.agents.ingestion_agent import IngestionAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.retrieval_agent import RetrievalAgent

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = None
    StateGraph = None


class SupportState(TypedDict, total=False):
    question: str
    file_ids: list[str]
    evidence: list[dict]
    reasoning: dict
    references: list[dict]
    retrieved_candidate_count: int


class SupportOrchestrator:
    def __init__(self) -> None:
        self.ingestion_agent = IngestionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.reasoning_agent = ReasoningAgent()
        self.citation_agent = CitationAgent()
        self.graph = self._build_graph()

    def preload(self) -> int:
        return self.ingestion_agent.preload_policies()

    def ingest_file(self, file_path, original_name: Optional[str] = None) -> dict:
        return self.ingestion_agent.ingest(file_path=file_path, original_name=original_name)

    def answer_question(self, question: str, file_ids: Optional[list[str]] = None) -> dict:
        started_at = perf_counter()
        if self.graph is not None:
            final_state = self.graph.invoke({"question": question, "file_ids": file_ids or []})
            reasoning = final_state["reasoning"]
            reasoning["references"] = final_state["references"]
            reasoning["runtime"] = self._runtime_metadata()
            reasoning["evaluation"] = self._evaluation_metrics(
                evidence=final_state.get("evidence", []),
                references=final_state.get("references", []),
                reasoning=reasoning,
                file_ids=file_ids,
                candidate_count=final_state.get("retrieved_candidate_count"),
                started_at=started_at,
            )
            return reasoning

        evidence = self.retrieval_agent.retrieve(question=question, file_ids=file_ids)
        reasoning = self.reasoning_agent.answer(question=question, evidence=evidence)
        references = self.citation_agent.format_references(evidence)
        reasoning["references"] = references
        reasoning["runtime"] = self._runtime_metadata()
        reasoning["evaluation"] = self._evaluation_metrics(
            evidence=evidence,
            references=references,
            reasoning=reasoning,
            file_ids=file_ids,
            candidate_count=len(evidence),
            started_at=started_at,
        )
        return reasoning

    def _build_graph(self):
        if StateGraph is None or END is None:
            return None

        graph_builder = StateGraph(SupportState)
        graph_builder.add_node("retrieve", self._retrieve_node)
        graph_builder.add_node("reason", self._reason_node)
        graph_builder.add_node("cite", self._cite_node)
        graph_builder.set_entry_point("retrieve")
        graph_builder.add_edge("retrieve", "reason")
        graph_builder.add_edge("reason", "cite")
        graph_builder.add_edge("cite", END)
        return graph_builder.compile()

    def _retrieve_node(self, state: SupportState) -> Dict[str, Any]:
        evidence = self.retrieval_agent.retrieve(
            question=state["question"],
            file_ids=state.get("file_ids") or None,
        )
        return {
            "evidence": evidence,
            "retrieved_candidate_count": len(evidence),
        }

    def _reason_node(self, state: SupportState) -> Dict[str, Any]:
        reasoning = self.reasoning_agent.answer(
            question=state["question"],
            evidence=state.get("evidence", []),
        )
        return {"reasoning": reasoning}

    def _cite_node(self, state: SupportState) -> Dict[str, Any]:
        references = self.citation_agent.format_references(state.get("evidence", []))
        return {"references": references}

    def _runtime_metadata(self) -> Dict[str, str]:
        return {
            "vector_backend": self.retrieval_agent.backend_name(),
            "llm_backend": self.reasoning_agent.backend_name(),
            "orchestration_backend": "langgraph" if self.graph is not None else "direct-fallback",
        }

    def _evaluation_metrics(
        self,
        *,
        evidence: list[dict],
        references: list[dict],
        reasoning: dict,
        file_ids: Optional[list[str]],
        candidate_count: Optional[int],
        started_at: float,
    ) -> Dict[str, Any]:
        scores = [float(item.get("score", 0.0)) for item in evidence if item.get("score") is not None]
        top_score = max(scores) if scores else None
        avg_score = round(sum(scores) / len(scores), 4) if scores else None

        return {
            "evidence_count": len(evidence),
            "retrieved_candidate_count": candidate_count if candidate_count is not None else len(evidence),
            "file_scope_applied": bool(file_ids),
            "top_relevance_score": round(top_score, 4) if top_score is not None else None,
            "average_relevance_score": avg_score,
            "grounded_response": bool(references) and len(evidence) > 0,
            "clarification_rate": 1.0 if reasoning.get("needs_clarification") else 0.0,
            "response_latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }
