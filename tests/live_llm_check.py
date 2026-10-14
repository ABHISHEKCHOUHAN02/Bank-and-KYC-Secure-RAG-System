"""
Real-API check for llm/client.py. Makes 3 tiny requests (list + plain + JSON), so it costs
almost nothing, but it DOES use your API key and quota. Not collected by pytest.

Run from the repo root:
    python tests/live_llm_check.py
    python tests/live_llm_check.py --model gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.schemas import GraderOutput  # noqa: E402
from llm.client import LLMError, LLMQuotaError, get_client  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model to test (default: GEMINI_MODEL)")
    args = ap.parse_args()
    model = args.model or settings.gemini_model
    client = get_client()
    ok = True

    def step(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label} {detail}".rstrip())

    try:
        models = client.list_models()
        step("key works, models listed", bool(models), f"({len(models)} models)")
        step(f"configured model '{model}' is in the list", model in models,
             "" if model in models else "-> pick a name from `python -m llm.client --list`")

        r = client.generate("live_plain", "Reply with the single word: ok", model=model, max_output_tokens=256)
        step("plain text reply", bool(r.text.strip()),
             f"reply={r.text.strip()!r} {r.call.latency_ms:.0f}ms in/out={r.call.input_tokens}/{r.call.output_tokens}")
        step("token counts reported", r.call.input_tokens is not None and r.call.output_tokens is not None)

        r = client.generate(
            "live_json",
            "Question: What is the minimum age to open a savings account in India?\n"
            "Passage: The bank sells credit cards with a 40-day interest-free period.\n"
            "Does the passage help answer the question?",
            schema=GraderOutput,
            model=model,
            max_output_tokens=1024,
        )
        step("JSON mode parses into GraderOutput", isinstance(r.parsed, GraderOutput),
             f"relevant={r.parsed.relevant} confidence={r.parsed.confidence}")
        step("irrelevant passage graded irrelevant (model quality, not a bug if it fails)",
             r.parsed.relevant is False)
    except LLMQuotaError as exc:
        print(f"[STOP] daily quota exhausted: {exc}")
        return 2
    except LLMError as exc:
        print(f"[FAIL] {exc}")
        return 1

    print("usage:", client.usage_summary())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())