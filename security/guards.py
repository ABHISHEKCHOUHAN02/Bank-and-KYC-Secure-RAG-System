"""Input and output safeguards for the banking RAG pipeline.

PII detection uses Presidio PatternRecognizer instances so the project can detect
Indian banking identifiers without downloading a general-purpose NLP model. Raw
matches are used only in memory to redact text; findings store offsets and types.
"""

from __future__ import annotations

import re
from functools import lru_cache

from app.schemas import (
    GuardResult,
    NumericGuardResult,
    PIIResult,
    PIIFinding,
    RefusalReason,
    RetrievedChunk,
)

INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.I),
    re.compile(r"\b(?:reveal|show|print|repeat)\s+(?:the\s+)?(?:system|developer)\s+prompt\b", re.I),
    re.compile(r"\b(?:developer|admin|jailbreak)\s+mode\b", re.I),
    re.compile(r"\b(?:bypass|override|disable)\s+(?:your\s+)?(?:rules|safety|guardrails|instructions)\b", re.I),
    re.compile(r"\bforget\s+(?:all\s+)?(?:your\s+)?(?:rules|instructions)\b", re.I),
)

ADVICE_PATTERNS = (
    re.compile(r"\bshould\s+i\s+(?:invest|borrow|take\s+(?:(?:a|an)\s+)?(?:(?:personal|home|student|business)\s+)?loan|apply\s+for\s+(?:(?:a|an)\s+)?(?:(?:personal|home|student|business)\s+)?loan|buy|sell|switch|close|prepay)\b", re.I),
    re.compile(r"\bwhich\s+(?:mutual\s+fund|stock|share|credit\s+card|loan|bank|investment)\b.{0,50}\b(?:best\s+for\s+me|should\s+i\s+choose|should\s+i\s+take)\b", re.I),
    re.compile(r"\bshould\s+i\s+(?:put|move)\s+(?:my\s+)?(?:money|savings|bonus)\b", re.I),
    re.compile(r"\bwhich\s+is\s+better\s+for\s+me\b", re.I),
    re.compile(r"\bis\s+this\s+bank\s+safe\s+enough\s+for\s+me\b", re.I),
    re.compile(r"\bis\s+(?:it|this)\s+(?:a\s+)?good\s+idea\s+for\s+me\s+to\s+(?:invest|borrow|take\s+(?:a\s+)?loan|close|switch)\b", re.I),
    re.compile(r"\b(?:tell|advise|recommend)\s+me\s+(?:what|where|which)\s+to\s+(?:invest|borrow|buy|choose)\b", re.I),
)


def check_prompt_injection(text: str) -> GuardResult:
    if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
        return GuardResult(allowed=False, reason=RefusalReason.INJECTION,
                           detail="The request contains an instruction override attempt.")
    return GuardResult(allowed=True)


def check_advice_scope(text: str) -> GuardResult:
    if any(pattern.search(text) for pattern in ADVICE_PATTERNS):
        return GuardResult(allowed=False, reason=RefusalReason.OUT_OF_SCOPE,
                           detail="The request asks for a personal financial recommendation.")
    return GuardResult(allowed=True)


@lru_cache(maxsize=1)
def _presidio_recognizers():
    from presidio_analyzer import Pattern, PatternRecognizer

    definitions = {
        "AADHAAR": [("aadhaar_12_digits", r"(?<!\d)[2-9](?:[ -]?\d){11}(?!\d)", 0.85)],
        "PAN": [("indian_pan", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.9)],
        "IFSC": [("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.9)],
        "PHONE": [("indian_mobile", r"(?<!\d)(?:\+91[ -]?)?[6-9](?:[ -]?\d){9}(?!\d)", 0.85)],
        "CARD": [("payment_card", r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)", 0.8)],
        "EMAIL": [("email_address", r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", 0.9)],
        "UPI_ID": [("upi_id", r"\b[A-Z0-9._-]{2,}@[A-Z][A-Z0-9-]{1,}\b", 0.75)],
    }
    return tuple(
        PatternRecognizer(
            supported_entity=entity,
            patterns=[Pattern(name=name, regex=regex, score=score) for name, regex, score in patterns],
        )
        for entity, patterns in definitions.items()
    )


ACCOUNT_RE = re.compile(
    r"\b(?:account(?:\s+(?:number|no\.?))?|acct\.?|a/c)\s*(?:is\s*)?[:#-]?\s*(?P<value>\d{8,18})\b",
    re.I,
)
DOB_RE = re.compile(
    r"\b(?:dob|date\s+of\s+birth|born)\s*(?:is\s*)?[:,-]?\s*(?P<value>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    re.I,
)
PERSON_RE = re.compile(r"\b(?:I\s+am|my\s+name\s+is)\s+(?P<value>[A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30}){1,2})\b")


def detect_pii(text: str) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    for recognizer in _presidio_recognizers():
        for result in recognizer.analyze(text=text, entities=[recognizer.supported_entities[0]]):
            findings.append(
                PIIFinding(entity_type=result.entity_type, start=result.start, end=result.end,
                           score=float(result.score))
            )
    for entity_type, pattern in (("ACCOUNT_NO", ACCOUNT_RE), ("DOB", DOB_RE), ("PERSON", PERSON_RE)):
        for match in pattern.finditer(text):
            span = match.span("value")
            findings.append(PIIFinding(entity_type=entity_type, start=span[0], end=span[1]))

    # Prefer a complete identifier when recognizers overlap (e.g. a 16-digit card
    # number could also contain a 10-digit mobile-shaped substring).
    priority = {"CARD": 8, "AADHAAR": 7, "PAN": 7, "IFSC": 7, "ACCOUNT_NO": 6,
                "PHONE": 5, "UPI_ID": 4, "EMAIL": 4, "DOB": 3, "PERSON": 2}
    chosen: list[PIIFinding] = []
    for item in sorted(findings, key=lambda f: (-(f.end - f.start), -priority.get(f.entity_type, 0), f.start)):
        if any(item.start < prior.end and prior.start < item.end for prior in chosen):
            continue
        chosen.append(item)
    return sorted(chosen, key=lambda f: f.start)


def redact_pii(text: str) -> PIIResult:
    findings = detect_pii(text)
    redacted = text
    for finding in reversed(findings):
        token = f"[REDACTED_{finding.entity_type}]"
        redacted = redacted[:finding.start] + token + redacted[finding.end:]
    return PIIResult(redacted_text=redacted, findings=findings)


NUMBER_RE = re.compile(r"(?<![\w])(?:₹\s*|INR\s*)?\d[\d,]*(?:\.\d+)?%?(?![\w])", re.I)
CITATION_RE = re.compile(r"\[\d+\]")


def _numbers(text: str) -> set[str]:
    return {
        re.sub(r"[^\d.]", "", match.group(0)).lstrip("0") or "0"
        for match in NUMBER_RE.finditer(text)
    }


def check_numeric_support(answer: str, retrieved: list[RetrievedChunk]) -> NumericGuardResult:
    answer_without_citations = CITATION_RE.sub("", answer)
    source_parts: list[str] = []
    for hit in retrieved:
        meta = hit.chunk.metadata
        source_parts.extend((hit.chunk.text, meta.circular_no or "", meta.date or "", meta.section or ""))
    supported = _numbers("\n".join(source_parts))
    unsupported = sorted(_numbers(answer_without_citations) - supported)
    return NumericGuardResult(passed=not unsupported, unsupported_numbers=unsupported)


def guard_answer(
    answer: str,
    retrieved: list[RetrievedChunk],
    *,
    redact_identifiers: bool = True,
) -> tuple[str, bool, NumericGuardResult]:
    """Redact any PII repeated by the model and reject unsupported numeric claims."""
    pii_result = redact_pii(answer) if redact_identifiers else PIIResult(redacted_text=answer)
    numeric_result = check_numeric_support(pii_result.redacted_text, retrieved)
    if not numeric_result.passed:
        return (
            "I can't provide this response because I couldn't verify every numeric claim against the retrieved documents.",
            bool(pii_result.findings),
            numeric_result,
        )
    return pii_result.redacted_text, bool(pii_result.findings), numeric_result
