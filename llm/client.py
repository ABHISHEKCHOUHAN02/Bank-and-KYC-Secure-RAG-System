"""
Single wrapper for Gemini calls: retries with backoff, free-tier rate spacing,
and latency/token logging (phase 8 needs those numbers).

Setup:
    pip install google-genai python-dotenv
    .env:  GEMINI_API_KEY=...
           GEMINI_MODEL=<current Flash model name>   # check the docs: old Flash versions get retired
           GEMINI_MIN_INTERVAL=4                     # seconds between calls on the free tier (optional)
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gemini-2.5-flash"  # fallback only: set GEMINI_MODEL and verify the name
RETRY_MARKERS = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE", "timed out")


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    retries: int = 0

    def add(self, r: LLMResult) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.latency_s += r.latency_s


USAGE = Usage()  # process-wide totals; read it after an eval run


def load_prompt(name: str) -> str:
    """Prompts live in prompts/<name>.md so ablations swap a file, not code."""
    return (ROOT / "prompts" / f"{name}.md").read_text(encoding="utf-8")


class LLMClient:
    def __init__(self, model: str | None = None, max_retries: int = 5) -> None:
        from google import genai

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY in your environment or .env")
        self._client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        self.min_interval = float(os.getenv("GEMINI_MIN_INTERVAL", "0"))
        self._last_call = 0.0

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> LLMResult:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
            response_mime_type="application/json" if json_mode else None,
        )
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                self._last_call = time.time()
                um = getattr(resp, "usage_metadata", None)
                result = LLMResult(
                    text=(resp.text or "").strip(),
                    prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
                    completion_tokens=getattr(um, "candidates_token_count", 0) or 0,
                    latency_s=time.time() - t0,
                    model=self.model,
                )
                USAGE.add(result)
                return result
            except Exception as exc:  # retry only on transient errors
                self._last_call = time.time()
                transient = any(m in str(exc) for m in RETRY_MARKERS)
                if not transient or attempt == self.max_retries:
                    raise
                USAGE.retries += 1
                time.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError("unreachable")

    def generate_json(self, prompt: str, system: str | None = None) -> dict:
        text = self.generate(prompt, system=system, json_mode=True).text
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                return json.loads(m.group(0))
            raise


@lru_cache(maxsize=1)
def get_client() -> LLMClient:
    return LLMClient()