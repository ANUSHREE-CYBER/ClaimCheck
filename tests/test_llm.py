"""Tests for the quota protection in llm.py (cache, budget, retries, throttle, fallback)."""

import json

import pytest
from pydantic import BaseModel

import config
from claimcheck import llm
from claimcheck.extract import ExtractResult


class Small(BaseModel):
    answer: str


class FakeProvider:
    """Stands in for Gemini/Groq. Returns or raises what we tell it, and counts calls."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, system, prompt, schema, max_output_tokens):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def live(monkeypatch):
    """Switch to 'live' mode with a fake Gemini key (no real call is ever made)."""
    monkeypatch.setattr(config, "MOCK_LLM", False)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")

    def install(provider_name, fake):
        monkeypatch.setitem(llm.PROVIDERS[provider_name], "call", fake)
        return fake
    return install


def ask(prompt="What?"):
    return llm.call_llm("test", "system", prompt, Small)


def test_mock_mode_uses_fixture_and_no_calls():
    result = llm.call_llm("extract", "s", "p", ExtractResult)
    assert len(result.claims) == 4
    assert llm.STATS["live_calls"] == 0


def test_second_identical_call_is_a_cache_hit(live):
    fake = live("gemini", FakeProvider('{"answer": "yes"}'))
    assert ask().answer == "yes"
    assert ask().answer == "yes"
    assert fake.calls == 1
    assert llm.STATS["live_calls"] == 1
    assert llm.STATS["cache_hits"] == 1


def test_bad_json_is_not_cached_and_not_retried(live):
    fake = live("gemini", FakeProvider("this is not json"))
    with pytest.raises(llm.BadOutput):
        ask()
    assert fake.calls == 1
    assert not (config.CACHE_DIR / "llm").exists()


def test_json_in_code_fences_is_accepted():
    assert llm.parse_json_output('```json\n{"answer": "ok"}\n```', Small).answer == "ok"


def test_at_most_two_retries_with_growing_wait(live, isolated):
    fake = live("gemini", FakeProvider(llm.TemporaryError("server busy")))
    with pytest.raises(llm.TemporaryError):
        ask()
    assert fake.calls == 3                    # 1 try + 2 retries
    assert isolated == [5, 15]                # growing wait between tries
    assert llm.STATS["live_calls"] == 3       # every retry counts


def test_retry_then_success(live):
    fake = live("gemini", FakeProvider(llm.RateLimited("slow down"), '{"answer": "ok"}'))
    assert ask().answer == "ok"
    assert fake.calls == 2


def test_no_retry_on_bad_request(live):
    fake = live("gemini", FakeProvider(llm.LLMError("400 bad request")))
    with pytest.raises(llm.LLMError):
        ask()
    assert fake.calls == 1


def test_no_retry_when_daily_quota_is_gone(live):
    fake = live("gemini", FakeProvider(llm.QuotaExhausted("daily quota used up")))
    with pytest.raises(llm.QuotaExhausted):
        ask()
    assert fake.calls == 1


def test_session_budget_stops_calls(live, monkeypatch):
    monkeypatch.setattr(config, "SESSION_CALL_CAP", 2)
    fake = live("gemini", FakeProvider('{"answer": "ok"}'))
    ask("one")
    ask("two")
    with pytest.raises(llm.BudgetExceeded):
        ask("three")
    assert fake.calls == 2


def test_daily_budget_is_saved_to_disk(live, monkeypatch):
    monkeypatch.setattr(config, "DAILY_CALL_CAP", 1)
    live("gemini", FakeProvider('{"answer": "ok"}'))
    ask("one")
    saved = json.loads((config.STATE_DIR / "usage.json").read_text())
    assert saved["calls"] == 1
    llm.STATS["live_calls"] = 0  # pretend the app restarted
    with pytest.raises(llm.BudgetExceeded):
        ask("two")


def test_cached_answers_still_work_when_budget_is_used_up(live, monkeypatch):
    live("gemini", FakeProvider('{"answer": "ok"}'))
    ask("demo")
    monkeypatch.setattr(config, "SESSION_CALL_CAP", 1)
    assert ask("demo").answer == "ok"   # cache hit, no budget needed


def test_warning_at_80_percent(monkeypatch):
    monkeypatch.setattr(config, "SESSION_CALL_CAP", 10)
    llm.STATS["live_calls"] = 7
    assert llm.budget_status()["warning"] is False
    llm.STATS["live_calls"] = 8
    assert llm.budget_status()["warning"] is True


def test_fallback_to_groq_when_gemini_quota_is_gone(live, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-groq-key")
    gemini = live("gemini", FakeProvider(llm.QuotaExhausted("daily quota used up")))
    groq = live("groq", FakeProvider('{"answer": "from groq"}'))
    assert ask().answer == "from groq"
    assert llm.STATS["last_provider"] == "groq"
    # Asking again: served from Groq's cache, no new calls to anyone.
    assert ask().answer == "from groq"
    assert (gemini.calls, groq.calls) == (1, 1)


def test_no_fallback_without_groq_key(live):
    live("gemini", FakeProvider(llm.QuotaExhausted("daily quota used up")))
    with pytest.raises(llm.QuotaExhausted, match="No fallback"):
        ask()


def test_throttle_waits_when_the_minute_is_full(live, monkeypatch, isolated):
    monkeypatch.setattr(config, "REQUESTS_PER_MINUTE", 2)
    live("gemini", FakeProvider('{"answer": "ok"}'))
    ask("one")
    ask("two")
    assert isolated == []
    ask("three")
    assert len(isolated) == 1 and 59 < isolated[0] <= 60


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.setattr(config, "MOCK_LLM", False)
    with pytest.raises(llm.ConfigError, match="GEMINI_API_KEY"):
        ask()


def test_http_errors_are_sorted_correctly():
    def kind(code, message):
        return type(llm._classify_http_error("Gemini", code, message))
    assert kind(429, "Quota GenerateRequestsPerDayPerProjectPerModel") is llm.QuotaExhausted
    assert kind(429, "Quota GenerateRequestsPerMinute") is llm.RateLimited
    assert kind(503, "unavailable") is llm.TemporaryError
    assert kind(400, "API key not valid") is llm.ConfigError
    assert kind(400, "bad schema") is llm.LLMError
