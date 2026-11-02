"""End-to-end tests of the pipeline in mock mode, with a fake Wikipedia."""

import requests

from claimcheck import llm, pipeline

FAKE_PAGES = {
    "Eiffel Tower": "The Eiffel Tower is a lattice tower on the Champ de Mars in Paris, France. "
                    "It was constructed as the centrepiece of the 1889 World's Fair.",
    "Gustave Eiffel": "He is best known for the Eiffel Tower, and his contribution to building "
                      "the Statue of Liberty in New York.",
}


def fake_search(query):
    if query not in FAKE_PAGES:
        return None
    return {"title": query, "url": "u", "text": FAKE_PAGES[query]}


def test_full_check_in_mock_mode(monkeypatch):
    monkeypatch.setattr(pipeline, "search_wikipedia", fake_search)
    result = pipeline.check_answer("any text")
    verdicts = [c["verdict"] for c in result["claims"]]
    assert verdicts == ["SUPPORTED", "CONTRADICTED", "SUPPORTED", "NOT_ENOUGH_EVIDENCE"]
    assert result["claims"][2]["reflected"] is True          # fixed by the reflect step
    assert result["claims"][2]["first_reason"]               # earlier reason kept for the UI
    assert all(c["quote_found"] for c in result["claims"][:3])
    assert result["reflect"] == {"candidates": 1, "rejudged": 1}
    assert result["live_calls"] == 0
    assert "contradicted" in result["trust_line"]


def test_reflect_skipped_when_new_search_finds_the_same_page(monkeypatch):
    monkeypatch.setattr(pipeline, "search_wikipedia", lambda q: fake_search("Eiffel Tower"))
    called = []
    monkeypatch.setattr(pipeline, "reflect_claims", lambda items: called.append(items) or [])
    result = pipeline.check_answer("any text")
    assert called == []                       # no extra LLM call was spent
    assert result["reflect"]["rejudged"] == 0


def test_budget_stop_during_reflect_keeps_first_results(monkeypatch):
    monkeypatch.setattr(pipeline, "search_wikipedia", fake_search)

    def out_of_budget(items):
        raise llm.BudgetExceeded("Call budget reached.")
    monkeypatch.setattr(pipeline, "reflect_claims", out_of_budget)
    result = pipeline.check_answer("any text")
    assert result["claims"][2]["verdict"] == "NOT_ENOUGH_EVIDENCE"
    assert any("skipped" in note for note in result["notes"])


def test_wikipedia_down_gives_not_enough_evidence(monkeypatch):
    def down(query):
        raise requests.ConnectionError()
    monkeypatch.setattr(pipeline, "search_wikipedia", down)
    result = pipeline.check_answer("any text")
    assert all(c["verdict"] == "NOT_ENOUGH_EVIDENCE" for c in result["claims"])
    assert result["notes"]


def test_trust_line_wording():
    def line(s, c, n):
        return pipeline.trust_line({"SUPPORTED": s, "CONTRADICTED": c, "NOT_ENOUGH_EVIDENCE": n})
    assert line(3, 0, 0) == "All 3 claims are supported by Wikipedia."
    assert "could not be checked" in line(1, 0, 2)
    assert line(0, 0, 0).startswith("No checkable")
