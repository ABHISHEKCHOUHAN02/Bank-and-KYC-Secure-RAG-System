"""
Offline tests for llm/client.py. No network, no API key: the SDK is replaced by a fake.

Run from the repo root:
    pytest
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import llm.client as client_mod
from app.schemas import GraderOutput
from llm.client import GeminiClient, LLMError, LLMQuotaError


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeAPIError(Exception):
    """Mimics an SDK error: has an HTTP-style `code` attribute."""

    def __init__(self, code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.code = code


def make_resp(text="ok", prompt=10, cand=5, thought=None, finish="STOP", block=None):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=cand,
            thoughts_token_count=thought,
        ),
        candidates=[SimpleNamespace(finish_reason=finish)],
        prompt_feedback=SimpleNamespace(block_reason=block),
    )


class FakeModels:
    """Plays back a script of responses/exceptions, one per generate_content call."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def list(self):
        return [
            SimpleNamespace(name="models/gemini-3.1-flash-lite", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/embed-x", supported_actions=["embedContent"]),
            SimpleNamespace(name="models/gemini-3.1-flash-lite", supported_actions=None),
        ]


@pytest.fixture
def sleeps(monkeypatch):
    rec: list[float] = []
    monkeypatch.setattr(client_mod, "_sleep", rec.append)
    return rec


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """Patch client settings: fast pacing, 3 retries, cache dir in tmp_path."""
    s = dataclasses.replace(
        client_mod.settings,
        llm_max_retries=3,
        llm_requests_per_minute=60000,
        llm_cache_enabled=False,
        llm_cache_dir=tmp_path / "cache",
        gemini_model="test-model",
    )
    monkeypatch.setattr(client_mod, "settings", s)
    return s


def make_client(script, rpm=60000):
    c = GeminiClient(api_key="fake", rpm=rpm)
    c._client = SimpleNamespace(models=FakeModels(script))
    return c


def api(c: GeminiClient) -> FakeModels:
    return c._client.models


# ----------------------------------------------------------------------
# Plain text + tokens
# ----------------------------------------------------------------------
def test_plain_text_and_tokens(cfg, sleeps):
    c = make_client([make_resp("hello", prompt=12, cand=7)])
    r = c.generate("generator", "hi")
    assert r.text == "hello"
    assert r.parsed is None and not r.cached
    assert (r.call.input_tokens, r.call.output_tokens) == (12, 7)
    assert r.call.name == "generator" and r.call.model == "test-model"
    assert c.calls == [r.call]


def test_thought_tokens_count_as_output(cfg, sleeps):
    c = make_client([make_resp("x", prompt=3, cand=5, thought=40)])
    assert c.generate("g", "hi").call.output_tokens == 45


def test_missing_usage_metadata(cfg, sleeps):
    resp = make_resp("x")
    resp.usage_metadata = None
    r = make_client([resp]).generate("g", "hi")
    assert (r.call.input_tokens, r.call.output_tokens) == (None, None)


def test_system_and_params_reach_sdk(cfg, sleeps):
    c = make_client([make_resp("x")])
    c.generate("g", "prompt", system="SYS", model="other-model", temperature=0.3, max_output_tokens=99)
    req = api(c).requests[0]
    assert req["model"] == "other-model" and req["contents"] == "prompt"
    assert req["config"].system_instruction == "SYS"
    assert req["config"].temperature == 0.3
    assert req["config"].max_output_tokens == 99


# ----------------------------------------------------------------------
# JSON mode + repair
# ----------------------------------------------------------------------
def test_json_mode_parses_and_requests_json(cfg, sleeps):
    c = make_client([make_resp('{"relevant": true, "confidence": 0.8, "reason": "r"}')])
    r = c.generate("grader", "q", schema=GraderOutput)
    assert isinstance(r.parsed, GraderOutput) and r.parsed.relevant is True
    cfg_used = api(c).requests[0]["config"]
    assert cfg_used.response_mime_type == "application/json"
    assert "JSON schema" in cfg_used.system_instruction


def test_json_fences_are_stripped(cfg, sleeps):
    c = make_client([make_resp('```json\n{"relevant": false}\n```')])
    assert c.generate("grader", "q", schema=GraderOutput).parsed.relevant is False


def test_json_repair_succeeds_on_second_try(cfg, sleeps):
    c = make_client([make_resp("not json at all"), make_resp('{"relevant": true}')])
    r = c.generate("grader", "q", schema=GraderOutput)
    assert r.parsed.relevant is True
    assert [x.name for x in c.calls] == ["grader", "grader_repair"]
    assert "not valid for the schema" in api(c).requests[1]["contents"]


def test_json_repair_gives_up(cfg, sleeps):
    c = make_client([make_resp("nope"), make_resp("still nope")])
    with pytest.raises(LLMError, match="not valid JSON"):
        c.generate("grader", "q", schema=GraderOutput)
    assert len(api(c).requests) == 2  # exactly one repair attempt


def test_schema_constraint_violation_triggers_repair(cfg, sleeps):
    c = make_client([make_resp('{"relevant": true, "confidence": 5}'), make_resp('{"relevant": true, "confidence": 0.5}')])
    assert c.generate("grader", "q", schema=GraderOutput).parsed.confidence == 0.5


# ----------------------------------------------------------------------
# Retries and error handling
# ----------------------------------------------------------------------
def test_retries_503_then_succeeds(cfg, sleeps):
    c = make_client([FakeAPIError(503, "unavailable"), FakeAPIError(429, "slow down"), make_resp("fine")])
    assert c.generate("g", "hi").text == "fine"
    assert len(api(c).requests) == 3
    backoffs = [s for s in sleeps if s >= 1.0]
    assert len(backoffs) == 2  # one backoff per failed attempt


def test_retries_exhausted(cfg, sleeps):
    c = make_client([FakeAPIError(503)] * 4)  # 1 try + 3 retries
    with pytest.raises(LLMError, match="Gemini call failed"):
        c.generate("g", "hi")
    assert len(api(c).requests) == 4


def test_non_retryable_error_fails_fast(cfg, sleeps):
    c = make_client([FakeAPIError(400, "bad request")])
    with pytest.raises(LLMError):
        c.generate("g", "hi")
    assert len(api(c).requests) == 1
    assert not [s for s in sleeps if s >= 1.0]


def test_404_adds_model_hint(cfg, sleeps):
    c = make_client([FakeAPIError(404, "model not found")])
    with pytest.raises(LLMError, match=r"--list"):
        c.generate("g", "hi")


def test_daily_quota_stops_immediately(cfg, sleeps):
    err = FakeAPIError(429, "Quota exceeded for metric: GenerateRequestsPerDayPerProjectPerModel")
    c = make_client([err, make_resp("never reached")])
    with pytest.raises(LLMQuotaError):
        c.generate("g", "hi")
    assert len(api(c).requests) == 1


def test_empty_reply_reports_reason(cfg, sleeps):
    c = make_client([make_resp(text=None, finish="MAX_TOKENS")])
    with pytest.raises(LLMError, match="MAX_TOKENS"):
        c.generate("g", "hi")


def test_missing_api_key():
    c = GeminiClient(api_key="", rpm=60)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        c._sdk()


# ----------------------------------------------------------------------
# Pacing
# ----------------------------------------------------------------------
def test_first_call_does_not_wait(cfg, sleeps):
    c = make_client([make_resp("x")], rpm=10)
    c.generate("g", "hi")
    assert sleeps == []  # the pacing bug: used to wait on the very first request


def test_second_call_waits_for_interval(cfg, sleeps):
    c = make_client([make_resp("a"), make_resp("b")], rpm=10)  # 6s interval
    c.generate("g", "1")
    c.generate("g", "2")
    assert len(sleeps) == 1 and 5.0 < sleeps[0] <= 6.0


# ----------------------------------------------------------------------
# Disk cache
# ----------------------------------------------------------------------
def test_cache_hit_skips_api(cfg, sleeps):
    c = make_client([make_resp("cached text", prompt=4, cand=2)])
    first = c.generate("g", "same prompt", use_cache=True)
    second = c.generate("g", "same prompt", use_cache=True)
    assert not first.cached and second.cached
    assert second.text == "cached text"
    assert (second.call.input_tokens, second.call.output_tokens) == (4, 2)
    assert len(api(c).requests) == 1


def test_cache_key_depends_on_model_and_prompt(cfg, sleeps):
    c = make_client([make_resp("a"), make_resp("b"), make_resp("c")])
    c.generate("g", "p", use_cache=True)
    c.generate("g", "p", model="other", use_cache=True)
    c.generate("g", "p2", use_cache=True)
    assert len(api(c).requests) == 3


def test_cache_off_by_default(cfg, sleeps):
    c = make_client([make_resp("a"), make_resp("b")])
    c.generate("g", "p")
    c.generate("g", "p")
    assert len(api(c).requests) == 2
    assert not cfg.llm_cache_dir.exists()


def test_cache_stores_json_text_and_reparses(cfg, sleeps):
    c = make_client([make_resp('{"relevant": true}')])
    c.generate("grader", "q", schema=GraderOutput, use_cache=True)
    r = c.generate("grader", "q", schema=GraderOutput, use_cache=True)
    assert r.cached and r.parsed.relevant is True
    files = list(cfg.llm_cache_dir.glob("*.json"))
    assert len(files) == 1 and "text" in json.loads(files[0].read_text(encoding="utf-8"))


def test_failed_json_is_not_cached(cfg, sleeps):
    c = make_client([make_resp("bad"), make_resp("bad again")])
    with pytest.raises(LLMError):
        c.generate("grader", "q", schema=GraderOutput, use_cache=True)
    assert not cfg.llm_cache_dir.exists() or not list(cfg.llm_cache_dir.glob("*.json"))


# ----------------------------------------------------------------------
# Bookkeeping
# ----------------------------------------------------------------------
def test_usage_summary_and_reset(cfg, sleeps):
    c = make_client([make_resp("a", prompt=10, cand=5), make_resp("b", prompt=20, cand=1, thought=4)])
    c.generate("g", "1")
    c.generate("g", "2")
    s = c.usage_summary()
    assert s["calls"] == 2 and s["input_tokens"] == 30 and s["output_tokens"] == 10
    c.reset_calls()
    assert c.usage_summary()["calls"] == 0


def test_list_models_filters_to_generate_content(cfg):
    c = make_client([])
    assert c.list_models() == ["gemini-3.1-flash-lite", "gemini-3.1-pro"]


class _Other(BaseModel):
    x: int


def test_cache_key_includes_schema(cfg):
    c = GeminiClient(api_key="k")
    a = c._cache_key("m", "s", "p", GraderOutput, 0.0, 10)
    b = c._cache_key("m", "s", "p", _Other, 0.0, 10)
    assert a != b