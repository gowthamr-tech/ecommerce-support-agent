from fastapi import APIRouter

from app.agents.orchestrator import SupportOrchestrator
from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])
orchestrator = SupportOrchestrator()


@router.post("/query", response_model=QueryResponse)
def query_support(payload: QueryRequest) -> QueryResponse:
    result = orchestrator.answer_question(question=payload.question, file_ids=payload.file_ids)
    return QueryResponse(**result)
