from typing import Literal, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    media_type: Literal["document", "image", "text"]
    extracted_summary: str
    chunks_indexed: int


class Reference(BaseModel):
    source: str
    chunk_id: str
    snippet: str
    score: Optional[float] = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    file_ids: Optional[list[str]] = None


class RuntimeMetadata(BaseModel):
    vector_backend: str
    llm_backend: str
    orchestration_backend: str


class EvaluationMetrics(BaseModel):
    evidence_count: int
    retrieved_candidate_count: int
    file_scope_applied: bool
    top_relevance_score: Optional[float] = None
    average_relevance_score: Optional[float] = None
    grounded_response: bool
    clarification_rate: float
    response_latency_ms: float


class QueryResponse(BaseModel):
    answer: str
    confidence: Literal["low", "medium", "high"]
    needs_clarification: bool
    references: list[Reference]
    reasoning_summary: str
    runtime: RuntimeMetadata
    evaluation: EvaluationMetrics


class HealthResponse(BaseModel):
    status: str


class CombinedResponse(BaseModel):
    uploads: list[UploadResponse]
    answer: str
    confidence: Literal["low", "medium", "high"]
    needs_clarification: bool
    references: list[Reference]
    reasoning_summary: str
    runtime: RuntimeMetadata
    evaluation: EvaluationMetrics
