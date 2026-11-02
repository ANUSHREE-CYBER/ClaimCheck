"""
Shared test setup. Every test runs in mock mode with its own temporary cache
and usage folders, and the internet is blocked, so tests never spend quota.
"""

import time
import types

import pytest
import requests

import config
from claimcheck import llm


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "MOCK_LLM", True)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    for key in llm.STATS:
        monkeypatch.setitem(llm.STATS, key, 0 if key in ("live_calls", "cache_hits") else None)
    llm._recent_call_times.clear()

    # Inside llm.py only: record waits instead of really sleeping (keeps tests fast).
    sleeps = []
    fake_time = types.SimpleNamespace(monotonic=time.monotonic,
                                      sleep=lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(llm, "time", fake_time)

    # Block the internet: any real HTTP request fails the test.
    def no_network(*args, **kwargs):
        raise AssertionError("A test tried to use the internet")
    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(requests, "post", no_network)
    return sleeps
