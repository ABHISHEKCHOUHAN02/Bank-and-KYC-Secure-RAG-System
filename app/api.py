"""FastAPI wrapper around the guarded BankGuard RAG pipeline."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.pipeline import answer_question
from app.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BankGuard RAG API",
    description="Evidence-grounded answers about banking and KYC policy.",
    version="0.1.0",
)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Liveness check; deliberately does not load the model or index."""
    return {"status": "ok", "service": "bankguard-rag"}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest) -> QueryResponse:
    """Run the complete security, retrieval, generation, and citation pipeline."""
    if not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="RAG service is not configured.")
    try:
        return answer_question(request.query)
    except Exception:
        # Do not log request content: it may contain sensitive customer information.
        logger.exception("RAG request failed")
        raise HTTPException(status_code=500, detail="The request could not be processed.") from None
