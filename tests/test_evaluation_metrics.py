from app.agents.orchestrator import SupportOrchestrator


def test_evaluation_metrics_are_computed() -> None:
    orchestrator = SupportOrchestrator.__new__(SupportOrchestrator)

    metrics = orchestrator._evaluation_metrics(
        evidence=[
            {"score": 0.4},
            {"score": 0.2},
        ],
        references=[{"source": "refund_policy.txt"}],
        reasoning={"needs_clarification": False},
        file_ids=["file-1"],
        candidate_count=2,
        started_at=0.0,
    )

    assert metrics["evidence_count"] == 2
    assert metrics["retrieved_candidate_count"] == 2
    assert metrics["file_scope_applied"] is True
    assert metrics["top_relevance_score"] == 0.4
    assert metrics["average_relevance_score"] == 0.3
    assert metrics["grounded_response"] is True
    assert metrics["clarification_rate"] == 0.0
    assert metrics["response_latency_ms"] >= 0.0


def test_evaluation_metrics_handle_empty_evidence() -> None:
    orchestrator = SupportOrchestrator.__new__(SupportOrchestrator)

    metrics = orchestrator._evaluation_metrics(
        evidence=[],
        references=[],
        reasoning={"needs_clarification": True},
        file_ids=None,
        candidate_count=0,
        started_at=0.0,
    )

    assert metrics["evidence_count"] == 0
    assert metrics["retrieved_candidate_count"] == 0
    assert metrics["file_scope_applied"] is False
    assert metrics["top_relevance_score"] is None
    assert metrics["average_relevance_score"] is None
    assert metrics["grounded_response"] is False
    assert metrics["clarification_rate"] == 1.0
    assert metrics["response_latency_ms"] >= 0.0


def test_resolve_file_scope_uses_latest_uploaded_file_for_uploaded_question() -> None:
    class DummySettings:
        class DummyPoliciesDir:
            def iterdir(self):
                return []

        policies_dir = DummyPoliciesDir()

    class DummyFileStore:
        def read(self):
            return [
                {"file_id": "policy-1", "filename": "refund_policy.txt", "media_type": "text"},
                {"file_id": "upload-1", "filename": "invoice.pdf", "media_type": "document"},
            ]

    orchestrator = SupportOrchestrator.__new__(SupportOrchestrator)
    orchestrator.settings = DummySettings()
    orchestrator.file_store = DummyFileStore()

    resolved = orchestrator._resolve_file_scope("I uploaded my invoice. Am I eligible for a refund?", None)

    assert resolved == ["upload-1"]
