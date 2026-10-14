"""User-facing callable pipeline for the Phase 4 RAG graph."""

from __future__ import annotations

import json
import logging
import sys
import time
from functools import lru_cache

from app.graph import build_graph
from app.schemas import (
    GuardResult,
    QueryRequest,
    QueryResponse,
    RefusalReason,
    Source,
    StepStatus,
    Trace,
    TraceStep,
)
from app.config import settings
from security.guards import check_advice_scope, check_prompt_injection, guard_answer, redact_pii
from cache.semantic import get_semantic_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_graph():
    """Reuse model, index, and compiled graph across requests in one process."""
    return build_graph()


def answer_question(question: str) -> QueryResponse:
    """Apply security checks, invoke the RAG graph, then guard its answer and citations."""
    request = QueryRequest(query=question)
    started = time.perf_counter()
    trace = Trace()
    trace.steps.append(
        TraceStep(
            name="ablation_profile",
            detail={
                "profile": settings.ablation_profile,
                "retrieval_mode": settings.retrieval_mode,
                "self_healing": settings.enable_self_healing,
                "web_fallback": settings.enable_web_fallback,
                "cache": settings.cache_enabled,
                "security": {
                    "injection_filter": settings.enable_injection_filter,
                    "pii_redaction": settings.enable_pii_redaction,
                    "scope_guard": settings.enable_scope_guard,
                    "output_guard": settings.enable_output_guard,
                },
            },
        )
    )

    def early_refusal(message: str, reason: RefusalReason) -> QueryResponse:
        trace.total_latency_ms = (time.perf_counter() - started) * 1000
        return QueryResponse(answer=message, trace=trace, refused=True, refusal_reason=reason)

    if settings.enable_injection_filter:
        guard = check_prompt_injection(request.query)
        trace.steps.append(TraceStep(name="injection_filter", status=StepStatus.OK if guard.allowed else StepStatus.BLOCKED,
                                     detail={"allowed": guard.allowed}))
        if not guard.allowed:
            return early_refusal(
                "I can help with banking policy questions, but I can't follow instructions that try to override the assistant's rules.",
                RefusalReason.INJECTION,
            )
    else:
        trace.steps.append(TraceStep(name="injection_filter", status=StepStatus.SKIPPED))

    if settings.enable_scope_guard:
        scope = check_advice_scope(request.query)
        trace.steps.append(TraceStep(name="scope_guard", status=StepStatus.OK if scope.allowed else StepStatus.BLOCKED,
                                     detail={"allowed": scope.allowed}))
        if not scope.allowed:
            return early_refusal(
                "I can explain relevant banking policies, but I can't recommend a personal financial decision.",
                RefusalReason.OUT_OF_SCOPE,
            )
    else:
        trace.steps.append(TraceStep(name="scope_guard", status=StepStatus.SKIPPED))

    if settings.enable_pii_redaction:
        pii_result = redact_pii(request.query)
        safe_question = pii_result.redacted_text
        pii_types = sorted({finding.entity_type for finding in pii_result.findings})
        trace.pii_types_found = pii_types
        trace.steps.append(TraceStep(name="pii_redaction", detail={"finding_count": len(pii_result.findings),
                                                                    "entity_types": pii_types}))
    else:
        safe_question = request.query
        trace.steps.append(TraceStep(name="pii_redaction", status=StepStatus.SKIPPED))

    cache_key_query = safe_question
    if settings.cache_enabled:
        cache_started = time.perf_counter()
        try:
            semantic_cache = get_semantic_cache()
            cache_hit = semantic_cache.lookup(cache_key_query)
            trace.steps.append(
                TraceStep(
                    name="semantic_cache_lookup",
                    latency_ms=(time.perf_counter() - cache_started) * 1000,
                    detail={"hit": cache_hit is not None,
                            "similarity": round(cache_hit.similarity, 4) if cache_hit else None},
                )
            )
            if cache_hit is not None:
                cached = QueryResponse.model_validate(cache_hit.payload)
                trace.cache_hit = True
                trace.total_latency_ms = (time.perf_counter() - started) * 1000
                return QueryResponse(
                    answer=cached.answer,
                    sources=cached.sources,
                    trace=trace,
                    refused=cached.refused,
                    refusal_reason=cached.refusal_reason,
                )
        except Exception as exc:
            logger.warning("Semantic cache lookup failed (%s); continuing without a cache hit.",
                           type(exc).__name__)
            trace.steps.append(
                TraceStep(name="semantic_cache_lookup", status=StepStatus.ERROR,
                          latency_ms=(time.perf_counter() - cache_started) * 1000,
                          detail={"hit": False, "error": type(exc).__name__})
            )
    else:
        trace.steps.append(TraceStep(name="semantic_cache_lookup", status=StepStatus.SKIPPED))

    state = get_graph().invoke(
        {
            "question": safe_question,
            "current_query": safe_question,
            "retrieved": [],
            "retries": 0,
            "web_used": False,
            "trace": trace,
        }
    )
    trace = state["trace"]
    trace.rewrites_used = state.get("retries", 0)
    trace.web_fallback_used = state.get("web_used", False)
    trace.total_latency_ms = (time.perf_counter() - started) * 1000

    answer = state["answer"]
    sources = state.get("sources", [])
    refused = state.get("refusal_reason") is not None
    refusal_reason = state.get("refusal_reason")
    if not refused and (settings.enable_output_guard or settings.enable_pii_redaction):
        answer, pii_redacted, numeric = guard_answer(
            answer,
            state.get("retrieved", []),
            redact_identifiers=settings.enable_pii_redaction,
        )
        output_findings = redact_pii(state["answer"]).findings if settings.enable_pii_redaction else []
        trace.pii_types_found = sorted(set(trace.pii_types_found) | {
            finding.entity_type for finding in output_findings
        })
        blocked = settings.enable_output_guard and not numeric.passed
        trace.steps.append(
            TraceStep(
                name="output_guard",
                status=StepStatus.BLOCKED if blocked else StepStatus.OK,
                detail={"passed": not blocked, "numeric_check_enabled": settings.enable_output_guard,
                        "pii_redacted": pii_redacted,
                        "unsupported_number_count": len(numeric.unsupported_numbers)},
            )
        )
        if blocked:
            sources = []
            refused = True
            refusal_reason = RefusalReason.OUTPUT_BLOCKED
    else:
        trace.steps.append(TraceStep(name="output_guard", status=StepStatus.SKIPPED))

    if settings.enable_pii_redaction and sources:
        safe_sources: list[Source] = []
        for source in sources:
            snippet_result = redact_pii(source.snippet)
            if snippet_result.findings:
                trace.pii_types_found = sorted(set(trace.pii_types_found) | {
                    finding.entity_type for finding in snippet_result.findings
                })
                source = source.model_copy(update={"snippet": snippet_result.redacted_text})
            safe_sources.append(source)
        sources = safe_sources

    response = QueryResponse(
        answer=answer,
        sources=sources,
        trace=trace,
        refused=refused,
        refusal_reason=refusal_reason,
    )
    # Only persist a response after PII redaction and numeric validation are on.
    if (settings.cache_enabled and settings.enable_pii_redaction and settings.enable_output_guard
            and not response.refused):
        cache_started = time.perf_counter()
        try:
            get_semantic_cache().store(cache_key_query, response)
            trace.steps.append(TraceStep(name="semantic_cache_store",
                                         latency_ms=(time.perf_counter() - cache_started) * 1000,
                                         detail={"stored": True}))
            trace.total_latency_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            logger.warning("Semantic cache store failed (%s).", type(exc).__name__)
            trace.steps.append(TraceStep(name="semantic_cache_store", status=StepStatus.ERROR,
                                         latency_ms=(time.perf_counter() - cache_started) * 1000,
                                         detail={"stored": False, "error": type(exc).__name__}))
            trace.total_latency_ms = (time.perf_counter() - started) * 1000
    return response


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: uv run python -m app.pipeline "Your banking policy question"')
    result = answer_question(" ".join(sys.argv[1:]))
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
