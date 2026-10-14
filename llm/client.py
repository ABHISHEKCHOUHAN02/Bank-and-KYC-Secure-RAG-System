"""
Single wrapper for every Gemini call in bank-rag.

Why one wrapper: pacing, retries, JSON parsing and the per-call log (latency, tokens) live
in one place, so the Phase 8 latency/cost numbers come from here and no other module has
to be instrumented.

    from llm.client import get_client
    client = get_client()
    res = client.generate("generator", prompt, system=SYSTEM)            # plain text
    res = client.generate("grader", prompt, schema=GraderOutput,          # validated JSON
                          model=settings.gemini_fast_model)
    res.text, res.parsed, res.call        # call is an app.schemas.LLMCall for the trace

Command line (from the repo root):
    python -m llm.client --list     # models your API key can call (verify names here)
    python -m llm.client --ping     # one tiny request: checks key, model name, latency

Notes
- JSON mode: asks for application/json and validates with the pydantic model yourself,
  with one automatic repair attempt. It does not use the SDK's native response_schema, to
  avoid schema-feature mismatches (defaults/constraints) between SDK versions.
- Thinking models: reasoning tokens are counted in `output_tokens` (they are billed as
  output, as far as I know) and can eat into max_output_tokens. If you see empty replies,
  raise max_output_tokens.
- The disk cache (LLM_CACHE=true) stores response text keyed by a hash of the request, not
  the prompt itself. Use it for evals only; leave it off in the live app.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.config import settings
from app.schemas import LLMCall

logger = logging.getLogger(__name__)

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
MAX_JSON_REPAIRS = 1

_sleep = time.sleep  # patched in tests


class LLMError(RuntimeError):
    """A Gemini call failed for good (bad key/model, blocked, empty, invalid JSON...)."""


class LLMQuotaError(LLMError):
    """Daily quota exhausted. Retrying will not help until it resets."""


@dataclass
class LLMResult:
    text: str
    call: LLMCall
    parsed: Any = None
    cached: bool = False


# ----------------------------------------------------------------------
# Error classification (heuristics: check them against real errors you hit)
# ----------------------------------------------------------------------
def _status_code(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_daily_quota(exc: Exception) -> bool:
    msg = re.sub(r"[\s_]", "", str(exc).lower())
    return "perday" in msg or "requestsperday" in msg


def _is_retryable(exc: Exception) -> bool:
    if _status_code(exc) in RETRYABLE_CODES:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ("unavailable", "overloaded", "timed out", "resource_exhausted"))


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)


# ----------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------
class GeminiClient:
    def __init__(self, api_key: str | None = None, rpm: int | None = None) -> None:
        self._api_key = api_key if api_key is not None else settings.gemini_api_key
        self._min_interval = 60.0 / max(rpm or settings.llm_requests_per_minute, 1)
        self._lock = threading.Lock()
        self._last_call: float | None = None  # None until the first request
        self._client = None
        self.calls: list[LLMCall] = []

    # --- plumbing -----------------------------------------------------
    def _sdk(self):
        if self._client is None:
            if not self._api_key:
                raise LLMError("GEMINI_API_KEY is not set. Put it in .env (see .env.example).")
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _wait_for_slot(self) -> None:
        with self._lock:
            if self._last_call is not None:
                wait = self._min_interval - (time.monotonic() - self._last_call)
                if wait > 0:
                    _sleep(wait)
            self._last_call = time.monotonic()

    def _cache_path(self, key: str):
        return settings.llm_cache_dir / f"{key}.json"

    def _cache_key(self, model, system, prompt, schema, temperature, max_tokens) -> str:
        blob = json.dumps(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "schema": schema.model_json_schema() if schema else None,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # --- one API call with retries -----------------------------------
    def _call_api(self, name: str, model: str, prompt: str, config):
        attempts = settings.llm_max_retries + 1
        for attempt in range(attempts):
            self._wait_for_slot()
            t0 = time.perf_counter()
            try:
                resp = self._sdk().models.generate_content(model=model, contents=prompt, config=config)
                return resp, (time.perf_counter() - t0) * 1000
            except LLMError:
                raise
            except Exception as exc:  # SDK errors vary by version: classify defensively
                if _is_daily_quota(exc):
                    raise LLMQuotaError(f"{name}: daily quota exhausted for {model}. {exc}") from exc
                last = attempt == attempts - 1
                if not _is_retryable(exc) or last:
                    hint = ""
                    if _status_code(exc) == 404:
                        hint = " Check GEMINI_MODEL / GEMINI_FAST_MODEL (python -m llm.client --list)."
                    raise LLMError(f"{name}: Gemini call failed ({type(exc).__name__}: {exc}).{hint}") from exc
                delay = min(60.0, 2.0 ** (attempt + 1)) + random.random()
                logger.warning("%s: %s; retry %d/%d in %.1fs", name, type(exc).__name__, attempt + 1, attempts - 1, delay)
                _sleep(delay)
        raise LLMError(f"{name}: unreachable")  # pragma: no cover

    @staticmethod
    def _extract(resp, name: str) -> str:
        text = getattr(resp, "text", None)
        if text:
            return text
        reason = None
        try:
            reason = resp.candidates[0].finish_reason
        except Exception:
            pass
        block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        raise LLMError(
            f"{name}: empty reply (finish_reason={reason}, block_reason={block}). "
            "If the reason is MAX_TOKENS, raise max_output_tokens."
        )

    @staticmethod
    def _tokens(resp) -> tuple[int | None, int | None]:
        um = getattr(resp, "usage_metadata", None)
        if um is None:
            return None, None
        tin = getattr(um, "prompt_token_count", None)
        cand = getattr(um, "candidates_token_count", None)
        thought = getattr(um, "thoughts_token_count", None)
        tout = None if cand is None and thought is None else (cand or 0) + (thought or 0)
        return tin, tout

    # --- public API ---------------------------------------------------
    def generate(
        self,
        name: str,
        prompt: str,
        *,
        system: str | None = None,
        schema=None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 2048,
        use_cache: bool | None = None,
    ) -> LLMResult:
        """
        name    label for the trace ("grader", "rewriter", "generator", ...)
        schema  optional pydantic model class: reply is requested as JSON and validated
        """
        from google.genai import types

        model = model or settings.gemini_model
        temperature = settings.llm_temperature if temperature is None else temperature
        use_cache = settings.llm_cache_enabled if use_cache is None else use_cache

        system_full = system
        if schema is not None:
            instruction = (
                "Reply with ONLY a JSON object that matches this JSON schema. "
                "No markdown fences, no commentary.\n" + json.dumps(schema.model_json_schema())
            )
            system_full = f"{system}\n\n{instruction}" if system else instruction

        cache_key = self._cache_key(model, system_full, prompt, schema, temperature, max_output_tokens) if use_cache else None
        if cache_key and self._cache_path(cache_key).exists():
            hit = json.loads(self._cache_path(cache_key).read_text(encoding="utf-8"))
            call = LLMCall(
                name=name,
                model=model,
                latency_ms=hit.get("latency_ms", 0.0),  # latency of the original call
                input_tokens=hit.get("input_tokens"),
                output_tokens=hit.get("output_tokens"),
            )
            self.calls.append(call)
            parsed = schema.model_validate_json(_strip_fences(hit["text"])) if schema else None
            return LLMResult(text=hit["text"], call=call, parsed=parsed, cached=True)

        cfg: dict[str, Any] = {"temperature": temperature, "max_output_tokens": max_output_tokens}
        if system_full:
            cfg["system_instruction"] = system_full
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
        config = types.GenerateContentConfig(**cfg)

        current_prompt, label = prompt, name
        for repair in range(MAX_JSON_REPAIRS + 1):
            resp, latency_ms = self._call_api(label, model, current_prompt, config)
            tin, tout = self._tokens(resp)
            call = LLMCall(name=label, model=model, latency_ms=latency_ms, input_tokens=tin, output_tokens=tout)
            self.calls.append(call)
            text = self._extract(resp, label)

            parsed = None
            if schema is not None:
                try:
                    parsed = schema.model_validate_json(_strip_fences(text))
                except ValueError as exc:
                    if repair < MAX_JSON_REPAIRS:
                        logger.warning("%s: invalid JSON, asking for a repair", name)
                        current_prompt = (
                            f"{prompt}\n\nYour previous reply was not valid for the schema "
                            f"({str(exc)[:200]}). Reply again with ONLY the JSON object."
                        )
                        label = f"{name}_repair"
                        continue
                    raise LLMError(f"{name}: reply is not valid JSON for {schema.__name__}: {exc}") from exc

            if cache_key:
                settings.llm_cache_dir.mkdir(parents=True, exist_ok=True)
                self._cache_path(cache_key).write_text(
                    json.dumps({"text": text, "latency_ms": latency_ms, "input_tokens": tin, "output_tokens": tout}),
                    encoding="utf-8",
                )
            return LLMResult(text=text, call=call, parsed=parsed)
        raise LLMError(f"{name}: unreachable")  # pragma: no cover

    # --- bookkeeping --------------------------------------------------
    def reset_calls(self) -> None:
        self.calls = []

    def usage_summary(self) -> dict:
        return {
            "calls": len(self.calls),
            "input_tokens": sum(c.input_tokens or 0 for c in self.calls),
            "output_tokens": sum(c.output_tokens or 0 for c in self.calls),
            "total_latency_ms": round(sum(c.latency_ms for c in self.calls), 1),
        }

    def list_models(self) -> list[str]:
        names = []
        for m in self._sdk().models.list():
            actions = getattr(m, "supported_actions", None)
            if actions is None or "generateContent" in actions:
                names.append(str(getattr(m, "name", "")).removeprefix("models/"))
        return sorted(names)


@lru_cache(maxsize=1)
def get_client() -> GeminiClient:
    return GeminiClient()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(description="Gemini client checks")
    ap.add_argument("--list", action="store_true", help="list models this key can call")
    ap.add_argument("--ping", action="store_true", help="send one tiny request")
    ap.add_argument("--model", default=None, help="model for --ping (default: GEMINI_MODEL)")
    args = ap.parse_args()

    c = get_client()
    if args.list:
        for n in c.list_models():
            print(n)
    if args.ping:
        r = c.generate("ping", "Reply with the single word: ok", model=args.model, max_output_tokens=256)
        print(f"reply={r.text.strip()!r} model={r.call.model} latency={r.call.latency_ms:.0f}ms "
              f"tokens in/out={r.call.input_tokens}/{r.call.output_tokens}")
    if not (args.list or args.ping):
        ap.print_help()