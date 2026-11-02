"""UI tests: run the real app.py with Streamlit's AppTest (mock mode, fake Wikipedia)."""

import pytest
from streamlit.testing.v1 import AppTest

import config
from claimcheck import llm, pipeline

APP = str(config.PROJECT_ROOT / "app.py")


def fake_search(query):
    pages = {
        "Eiffel Tower": "The Eiffel Tower is a lattice tower on the Champ de Mars in Paris, France. "
                        "It was constructed as the centrepiece of the 1889 World's Fair.",
        "Gustave Eiffel": "He is known for his contribution to building the Statue of Liberty.",
    }
    return {"title": query, "url": "u", "text": pages[query]} if query in pages else None


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(pipeline, "search_wikipedia", fake_search)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def all_markdown(at):
    return " ".join(m.value for m in at.markdown)


def test_page_loads_with_mock_badge_and_tabs(app):
    assert "MOCK MODE" in app.error[0].value
    assert [t.label for t in app.tabs] == ["Check an answer", "Evaluation", "How it works"]
    assert any("Live calls used" in m.value for m in app.sidebar.markdown)


def test_evaluation_tab_shows_saved_results(app):
    assert "Overall accuracy" in all_markdown(app)
    assert "small test set" in all_markdown(app)


def test_check_button_shows_colour_coded_claims(app):
    app.text_area(key="answer_text").input("The Eiffel Tower was built for the 1900 World's Fair.").run()
    app.button[-1].click().run()  # the "Check" button
    assert not app.exception
    text = all_markdown(app)
    assert "Contradicted" in text and "Supported" in text and "Proof:" in text
    assert "Re-checked" in text  # the reflect step is visible
    assert [m.value for m in app.metric][:3] == ["2", "1", "1"]


def test_demo_button_loads_text_and_runs(app):
    app.button(key="demo_0").click().run()
    assert not app.exception
    assert "Taj Mahal" in app.text_area(key="answer_text").value
    assert "Claim by claim" in [s.value for s in app.subheader]


def test_empty_text_gives_a_hint(app):
    app.button[-1].click().run()
    assert any("Paste an answer first" in w.value for w in app.warning)


def test_budget_stop_is_a_friendly_message(app, monkeypatch):
    monkeypatch.setattr(config, "MOCK_LLM", False)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "SESSION_CALL_CAP", 0)
    app.text_area(key="answer_text").input("Some new text.").run()
    app.button[-1].click().run()
    assert not app.exception
    assert any("Call budget reached" in e.value for e in app.error)


def test_quota_gone_is_a_friendly_message(app, monkeypatch):
    monkeypatch.setattr(config, "MOCK_LLM", False)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")

    def quota_gone(*args):
        raise llm.QuotaExhausted("Gemini's free daily quota is used up.")
    monkeypatch.setitem(llm.PROVIDERS["gemini"], "call", quota_gone)
    app.text_area(key="answer_text").input("Some new text.").run()
    app.button[-1].click().run()
    assert not app.exception
    assert any("daily quota" in e.value and "demo examples" in e.value for e in app.error)
