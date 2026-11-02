"""Tests for the Wikipedia tool (with a fake Wikipedia, no internet)."""

import pytest
import requests

from claimcheck import evidence


class FakeResponse:
    def __init__(self, data, status_code=200):
        self.data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self.data


def fake_wikipedia(monkeypatch, data, statuses=(200,)):
    """Fake Wikipedia. 'statuses' = HTTP status for each request in turn (last one repeats)."""
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append(params["gsrsearch"])
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        return FakeResponse(data, status)
    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def test_trim_text_ends_at_a_sentence():
    text = "First sentence here. " * 100
    trimmed = evidence.trim_text(text, max_chars=100)
    assert len(trimmed) <= 100 and trimmed.endswith(".")


def test_short_text_is_only_tidied():
    assert evidence.trim_text("Short.  Text.") == "Short. Text."


def test_search_returns_title_url_text_and_caches(monkeypatch):
    data = {"query": {"pages": {"1": {"title": "Mount Everest",
                                      "extract": "Mount Everest is Earth's highest mountain."}}}}
    calls = fake_wikipedia(monkeypatch, data)
    first = evidence.search_wikipedia("Mount Everest")
    second = evidence.search_wikipedia("Mount Everest")
    assert first == second
    assert first["url"] == "https://en.wikipedia.org/wiki/Mount_Everest"
    assert calls == ["Mount Everest"]  # second time came from the disk cache


def test_too_many_requests_is_retried(monkeypatch, isolated):
    data = {"query": {"pages": {"1": {"title": "Nile", "extract": "The Nile is a river."}}}}
    calls = fake_wikipedia(monkeypatch, data, statuses=(429, 200))
    assert evidence.search_wikipedia("Nile")["title"] == "Nile"
    assert len(calls) == 2 and 3 in isolated


def test_wikipedia_gives_up_after_two_retries(monkeypatch):
    calls = fake_wikipedia(monkeypatch, {}, statuses=(429,))
    with pytest.raises(requests.HTTPError):
        evidence.search_wikipedia("Nile")
    assert len(calls) == 3


def test_no_result_returns_none(monkeypatch):
    fake_wikipedia(monkeypatch, {"batchcomplete": ""})
    assert evidence.search_wikipedia("Zorblax Quantum Paper 2031") is None
