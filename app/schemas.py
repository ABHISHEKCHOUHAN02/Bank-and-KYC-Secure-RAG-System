"""
Shared data models for bank-rag.

Pydantic models are used for anything that crosses a boundary (API, LLM JSON
output, logs). The LangGraph state is a TypedDict, which is the pattern LangGraph
documents for state schemas (verify against the version you install).

Privacy rule: no model here stores raw PII. Findings keep entity type and
character offsets only, never the matched value.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

# Upper bound on user query length. Consider moving to config.py if you tune it.
MAX_QUERY_CHARS = 2000


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------
class SourceType(str, Enum):
    RBI = "rbi"
    FICTIONAL_BANK = "fictional_bank"
    WEB = "web"


class RefusalReason(str, Enum):
    INJECTION = "prompt_injection"
    OUT_OF_SCOPE = "out_of_scope_advice"  # "should I invest / take a loan"
    NO_EVIDENCE = "insufficient_evidence"  # grader failed after retries
    OUTPUT_BLOCKED = "output_guard_blocked"


class StepStatus(str, Enum):
    OK = "ok"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    ERROR = "error"


# ----------------------------------------------------------------------
# Documents and retrieval
# ----------------------------------------------------------------------
class ChunkMetadata(BaseModel):
    doc: str  # file or document title
    source_type: SourceType = SourceType.RBI
    circular_no: str | None = None
    date: str | None = None  # keep as ISO string; circular dates are often messy
    section: str | None = None
    page: int | None = None
    url: str | None = None  # for web results


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    retriever: Literal["dense", "bm25", "rrf", "web"] = "rrf"


# ----------------------------------------------------------------------
# LLM structured outputs
# ----------------------------------------------------------------------
class GraderOutput(BaseModel):
    """JSON the grader LLM must return."""
    # it must be valid JSON, so we can't use a bool for `relevant` because the LLM might return "true" or "false" as a string. Instead, we use a float confidence score and a reason string.

    relevant: bool
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""
    missing_info: str = ""  # hint the rewriter can use


class RewriterOutput(BaseModel):
    rewritten_query: str = Field(min_length=1)


# ----------------------------------------------------------------------
# Security results
# ----------------------------------------------------------------------
class PIIFinding(BaseModel):
    entity_type: str  # e.g. AADHAAR, PAN, IFSC, UPI_ID, CARD, PHONE
    start: int
    end: int
    score: float = 1.0
    # Deliberately no `value` field: never keep the raw PII.


class GuardResult(BaseModel):
    """Generic result for injection filter / scope guard."""

    allowed: bool
    reason: RefusalReason | None = None
    detail: str = ""


class PIIResult(BaseModel):
    redacted_text: str
    findings: list[PIIFinding] = Field(default_factory=list)


class NumericGuardResult(BaseModel):
    passed: bool
    # Numbers present in the answer but not found in the retrieved context.
    unsupported_numbers: list[str] = Field(default_factory=list)


class OutputGuardResult(BaseModel):
    passed: bool
    final_answer: str
    pii_redacted: bool = False
    numeric: NumericGuardResult | None = None
    disclaimer_added: bool = False


# ----------------------------------------------------------------------
# Tracing and cost
# ----------------------------------------------------------------------
class LLMCall(BaseModel):
    """One record per Gemini call, written by llm/client.py."""

    name: str  # "grader", "rewriter", "generator", ...
    model: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class TraceStep(BaseModel):
    name: str  # "injection_filter", "retrieve", "grade", ...
    status: StepStatus = StepStatus.OK
    latency_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)  # keep free of raw PII


class Trace(BaseModel):
    steps: list[TraceStep] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    cache_hit: bool = False
    rewrites_used: int = 0
    web_fallback_used: bool = False
    pii_types_found: list[str] = Field(default_factory=list)  # types only
    total_latency_ms: float = 0.0


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class Source(BaseModel):
    """A citation shown to the user."""

    doc: str
    source_type: SourceType
    circular_no: str | None = None
    date: str | None = None
    section: str | None = None
    page: int | None = None
    url: str | None = None
    snippet: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    trace: Trace = Field(default_factory=Trace)
    refused: bool = False
    refusal_reason: RefusalReason | None = None


# ----------------------------------------------------------------------
# LangGraph state
# ----------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    """
    State passed between graph nodes. total=False because nodes fill it in
    progressively (e.g. `answer` does not exist until the generator runs).
    """

    question: str  # already PII-redacted
    current_query: str  # may be rewritten
    retrieved: list[RetrievedChunk]
    grade: GraderOutput
    retries: int
    web_used: bool
    answer: str
    sources: list[Source]
    trace: Trace
    refusal_reason: RefusalReason | None