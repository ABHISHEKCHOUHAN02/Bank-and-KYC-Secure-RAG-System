"""Answer generation over retrieved chunks (the plain-RAG baseline uses this as is)."""
from __future__ import annotations

from llm.client import LLMResult, get_client, load_prompt


def format_context(hits) -> str:
    """hits: list[RetrievedChunk]. Numbered so the model can cite [1], [2]..."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        m = h.chunk.metadata
        head = " | ".join(x for x in (m.doc, m.circular_no, m.section) if x)
        blocks.append(f"[{i}] {head}\n{h.chunk.text}")
    return "\n\n".join(blocks)


def generate_answer(query: str, hits) -> dict:
    prompt = (
        load_prompt("generator")
        .replace("{context}", format_context(hits))
        .replace("{question}", query)
    )
    r: LLMResult = get_client().generate(prompt)
    return {
        "answer": r.text,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "latency_s": r.latency_s,
    }