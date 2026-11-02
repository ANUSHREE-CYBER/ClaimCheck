"""Tests for the Wikipedia tool (with a fake Wikipedia, no internet)."""

import requests

from claimcheck import evidence


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def fake_wikipedia(monkeypatch, data):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append(params["gsrsearch"])
        return FakeResponse(data)
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


def test_no_result_returns_none(monkeypatch):
    fake_wikipedia(monkeypatch, {"batchcomplete": ""})
    assert evidence.search_wikipedia("Zorblax Quantum Paper 2031") is None
