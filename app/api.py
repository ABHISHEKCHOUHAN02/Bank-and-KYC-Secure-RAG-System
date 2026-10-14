"""FastAPI wrapper around the guarded BankGuard RAG pipeline."""

from __future__ import annotations

import logging
import secrets
from typing import Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from app.config import settings
from app.documents import (
    INDEX_LOCK,
    MAX_UPLOAD_BYTES,
    DocumentOperationError,
    IndexRebuildError,
    delete_document,
    list_documents,
    upload_document,
)
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
        with INDEX_LOCK:
            return answer_question(request.query)
    except Exception:
        # Do not log request content: it may contain sensitive customer information.
        logger.exception("RAG request failed")
        raise HTTPException(status_code=500, detail="The request could not be processed.") from None


def _authorize_document_admin(token: str) -> None:
    if not settings.docs_admin_token:
        raise HTTPException(status_code=503, detail="Document management is not configured.")
    if not token or not secrets.compare_digest(token, settings.docs_admin_token):
        raise HTTPException(status_code=401, detail="Invalid document-management token.")


@app.get("/documents", tags=["documents"])
def get_documents(x_admin_token: str = Header(default="")) -> list[dict]:
    """List source files; upload and delete controls require the admin token."""
    _authorize_document_admin(x_admin_token)
    return list_documents()


@app.post("/documents", tags=["documents"])
async def add_document(
    file: UploadFile = File(...),
    source_type: Literal["rbi", "fictional_bank"] = Form(...),
    x_admin_token: str = Header(default=""),
) -> dict:
    """Upload a PDF/TXT/MD source and rebuild all retrieval indexes."""
    _authorize_document_admin(x_admin_token)
    filename = file.filename or ""
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    try:
        return upload_document(filename, source_type, content)
    except DocumentOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IndexRebuildError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/documents/{source_type}/{filename}", tags=["documents"])
def remove_document(
    source_type: Literal["rbi", "fictional_bank"],
    filename: str,
    x_admin_token: str = Header(default=""),
) -> dict:
    """Remove an uploaded source and rebuild all retrieval indexes."""
    _authorize_document_admin(x_admin_token)
    try:
        return delete_document(filename, source_type)
    except DocumentOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IndexRebuildError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
