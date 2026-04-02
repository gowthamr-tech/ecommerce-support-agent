from __future__ import annotations

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
        if self.graph is not None:
            final_state = self.graph.invoke({"question": question, "file_ids": file_ids or []})
            reasoning = final_state["reasoning"]
            reasoning["references"] = final_state["references"]
            reasoning["runtime"] = self._runtime_metadata()
            return reasoning

        evidence = self.retrieval_agent.retrieve(question=question, file_ids=file_ids)
        reasoning = self.reasoning_agent.answer(question=question, evidence=evidence)
        references = self.citation_agent.format_references(evidence)
        reasoning["references"] = references
        reasoning["runtime"] = self._runtime_metadata()
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
        return {"evidence": evidence}

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
