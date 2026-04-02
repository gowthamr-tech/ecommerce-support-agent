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
