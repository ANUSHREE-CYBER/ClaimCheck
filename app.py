"""
app.py - the ClaimCheck web page (Streamlit).

Run with:   streamlit run app.py

The page only shows things. All the real work happens in claimcheck/pipeline.py.
"""

import html
import json

import streamlit as st

import config
from claimcheck import llm, prompts
from claimcheck.judge import CONTRADICTED, NOT_ENOUGH, SUPPORTED
from claimcheck.pipeline import check_answer

PRINCIPLE = "Don't trust an AI answer as a whole. Check it claim by claim, and show the proof."

# How each verdict looks on screen: (label, icon, colour)
VERDICT_STYLE = {
    SUPPORTED: ("Supported", "✅", "#1f9d55"),
    CONTRADICTED: ("Contradicted", "❌", "#d64545"),
    NOT_ENOUGH: ("Not enough evidence", "❔", "#8a8f98"),
}


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


DEMO_CASES = load_json(config.PROJECT_ROOT / "demo" / "demo_cases.json") or []


# ---------------------------------------------------------------------------
# Sidebar: mode, model, and the call counter (always visible)
# ---------------------------------------------------------------------------

def show_sidebar():
    st.sidebar.title("🔎 ClaimCheck")
    if config.MOCK_LLM:
        st.sidebar.error("**MOCK MODE**\n\nThe AI is replaced by fake, ready-made answers. "
                         "These are NOT real results. Set MOCK_LLM=0 in .env for real checks.")
        model = "mock (no AI calls)"
    else:
        model = config.GEMINI_MODEL
        if config.GROQ_API_KEY:
            model += f"  \n(fallback: Groq {config.GROQ_MODEL})"

    status = llm.budget_status()
    st.sidebar.markdown(f"**Model:** {model}")
    st.sidebar.markdown(f"**Live calls used:** {status['session_used']} / {status['session_cap']} (this run)")
    st.sidebar.markdown(f"**Live calls today:** {status['daily_used']} / {status['daily_cap']}")
    st.sidebar.markdown(f"**Cache hits:** {status['cache_hits']}")
    if status["warning"]:
        st.sidebar.warning("Over 80% of the call budget is used. New text may stop soon; "
                           "cached examples will keep working.")
    st.sidebar.caption("Only real API requests count as live calls. Answers already in the "
                       "cache are free. Max 3 live calls per check.")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def claim_card(c):
    """One coloured card per claim: verdict, claim, reason, proof."""
    label, icon, colour = VERDICT_STYLE[c["verdict"]]
    parts = [
        f'<div style="border-left: 6px solid {colour}; background: {colour}1a; '
        f'padding: 0.7rem 1rem; border-radius: 6px; margin-bottom: 0.8rem;">',
        f'<div style="color: {colour}; font-weight: 700;">{icon} {label}</div>',
        f'<div style="font-size: 1.1rem; margin: 0.2rem 0 0.4rem;">{html.escape(c["claim"])}</div>',
        f'<div><b>Why:</b> {html.escape(c["reason"])}</div>',
    ]
    if c["evidence_quote"]:
        source = c["evidence"][-1] if c["evidence"] else None
        for ev in c["evidence"]:  # name the page the quote really came from
            if c["evidence_quote"].lower()[:30] in ev["text"].lower():
                source = ev
        link = f' (<a href="{source["url"]}" target="_blank">Wikipedia: {html.escape(source["title"])}</a>)' if source else ""
        parts.append(f'<div style="margin-top: 0.4rem;"><b>Proof:</b> <i>"{html.escape(c["evidence_quote"])}"</i>{link}</div>')
        if not c["quote_found"]:
            parts.append('<div style="color: #b7791f;">⚠ This quote was not found word-for-word '
                         'in the evidence, so check it yourself.</div>')
    if c.get("reflected"):
        parts.append(f'<div style="margin-top: 0.4rem; opacity: 0.85;">🔁 <b>Re-checked:</b> the first search '
                     f'was not enough ("{html.escape(c["first_reason"])}"), so ClaimCheck searched again.</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

    if c["evidence"]:
        with st.expander("Evidence the judge saw"):
            for ev in c["evidence"]:
                st.markdown(f"**[{ev['title']}]({ev['url']})**")
                st.caption(ev["text"])
    else:
        st.caption("No Wikipedia page was found for this claim.")


def show_result(result):
    counts = result["counts"]
    if counts[CONTRADICTED]:
        st.error(result["trust_line"])
    elif counts[SUPPORTED] and counts[SUPPORTED] == sum(counts.values()):
        st.success(result["trust_line"])
    else:
        st.warning(result["trust_line"])

    cols = st.columns(5)
    cols[0].metric("✅ Supported", counts[SUPPORTED])
    cols[1].metric("❌ Contradicted", counts[CONTRADICTED])
    cols[2].metric("❔ Not enough evidence", counts[NOT_ENOUGH])
    cols[3].metric("Live calls used", result["live_calls"])
    cols[4].metric("From cache", result["cache_hits"])

    for note in result["notes"]:
        st.info(note)

    if not result["claims"]:
        return
    st.subheader("Claim by claim")
    for c in result["claims"]:
        claim_card(c)
    st.caption("A claim with no Wikipedia proof is marked 'Not enough evidence', not 'false'. "
               "The AI judge can also be wrong, which is why every verdict shows its proof.")


# ---------------------------------------------------------------------------
# Running a check (with friendly errors)
# ---------------------------------------------------------------------------

def run_check(text):
    if not text.strip():
        st.warning("Paste an answer first.")
        return
    try:
        with st.spinner("Checking claim by claim… (extract → search Wikipedia → judge → re-check)"):
            st.session_state["result"] = check_answer(text)
    except llm.BudgetExceeded as error:
        st.error(f"⛔ {error}")
    except (llm.QuotaExhausted, llm.RateLimited) as error:
        st.error(f"⏳ {error}  \nTip: the demo examples above are saved and still work.")
    except llm.LLMError as error:
        st.error(f"Something went wrong with the AI step: {error}")
    except Exception as error:  # anything unexpected: say it plainly, do not crash the page
        st.error(f"Unexpected problem ({type(error).__name__}): {error}")


def load_demo(index):
    """Button callback: put a demo answer in the text box and check it."""
    st.session_state["answer_text"] = DEMO_CASES[index]["text"]
    st.session_state["run_now"] = True


def check_tab():
    st.markdown(f"#### *{PRINCIPLE}*")
    st.write("Paste an answer written by an AI. ClaimCheck splits it into single facts, "
             "looks each one up on Wikipedia, and shows what holds up.")

    if DEMO_CASES:
        st.markdown("**Try an example:**")
        cols = st.columns(len(DEMO_CASES))
        for i, case in enumerate(DEMO_CASES):
            cols[i].button(case["name"], key=f"demo_{i}", help=case["what_it_shows"],
                           on_click=load_demo, args=(i,), width="stretch")

    text = st.text_area("AI answer to check", key="answer_text", height=150,
                        placeholder="e.g. The Eiffel Tower is in Paris. It was finished in 1925.")
    clicked = st.button("Check", type="primary")

    if clicked or st.session_state.pop("run_now", False):
        st.session_state.pop("result", None)
        run_check(text)

    if "result" in st.session_state:
        show_result(st.session_state["result"])


# ---------------------------------------------------------------------------
# Other tabs
# ---------------------------------------------------------------------------

def eval_tab():
    results = load_json(config.PROJECT_ROOT / "eval" / "results.json")
    if not results:
        st.info("The evaluation has not been run yet (eval/run_eval.py).")
        return
    st.markdown(results.get("summary_markdown", ""))
    with st.expander("Every labelled claim and what ClaimCheck said"):
        st.dataframe(results.get("claim_rows", []), width="stretch")


def how_tab():
    st.markdown("""
**The loop: plan → act → judge → reflect** (max 3 live AI calls per check)

| Step | What happens | AI calls |
|---|---|---|
| A. Extract (plan) | Split the answer into up to 8 short, self-contained claims, each with a Wikipedia search query | 1 |
| B. Evidence (act) | Search Wikipedia for each claim and keep ~1000 characters of the best page's intro | 0 |
| C. Judge | All claims judged in one batch: reason first, then verdict, then a short proof quote | 1 |
| D. Reflect | Unclear claims (max 3) are searched again with a better query and re-judged once | 0 or 1 |

**Quota protection:** disk cache (same question = 0 calls), mock mode for development,
call budget with this counter, max 2 retries for temporary errors only, and a throttle of
""" + f"{config.REQUESTS_PER_MINUTE} requests per minute." + """

**Checks done in plain code (not by the AI):** a claim with no evidence cannot be marked
true or false; every proof quote is checked to really appear in the Wikipedia text.
""")
    st.subheader("The prompts (from prompts.py)")
    st.caption(f"Versions: {prompts.PROMPT_VERSIONS}")
    with st.expander("1. Extract: few-shot + structured output"):
        st.code(prompts.EXTRACT_SYSTEM, language="text")
        st.code(prompts.EXTRACT_EXAMPLES, language="text")
    with st.expander("2. Judge: zero-shot + chain-of-thought (reason before verdict)"):
        st.code(prompts.JUDGE_SYSTEM, language="text")
    with st.expander("3. Reflect: second look with new evidence"):
        st.code(prompts.REFLECT_SYSTEM, language="text")


# ---------------------------------------------------------------------------

st.set_page_config(page_title="ClaimCheck", page_icon="🔎", layout="wide")
st.title("🔎 ClaimCheck")
if config.MOCK_LLM:
    st.error("**MOCK MODE:** results below are fake, ready-made answers for testing, not real checks.")

tab_check, tab_eval, tab_how = st.tabs(["Check an answer", "Evaluation", "How it works"])
with tab_check:
    check_tab()
with tab_eval:
    eval_tab()
with tab_how:
    how_tab()

# Drawn last so the counter includes the check that just ran.
show_sidebar()
