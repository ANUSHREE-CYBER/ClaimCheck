"""
judge.py - Step C (judge) and Step D (reflect).

Judge:   all claims + their evidence go to the model in ONE call.
Reflect: claims that came back NOT_ENOUGH_EVIDENCE are re-judged once with
         new evidence, again in ONE call. Never more than one round.

After the model answers, plain Python double-checks it:
  - every claim gets a verdict (a missing one becomes NOT_ENOUGH_EVIDENCE)
  - a claim with no evidence at all cannot be SUPPORTED or CONTRADICTED
  - the proof quote must be short and really appear in the evidence
"""

from typing import Literal

from pydantic import BaseModel

import config
from claimcheck import prompts
from claimcheck.llm import call_llm

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_ENOUGH = "NOT_ENOUGH_EVIDENCE"


# The JSON shape we expect back. Field order matters: "reason" comes before
# "verdict", so the model explains first and decides second (chain-of-thought).
class ClaimVerdict(BaseModel):
    claim_id: int
    reason: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"]
    evidence_quote: str
    better_search_query: str


class JudgeResult(BaseModel):
    results: list[ClaimVerdict]


def _normalise(text):
    """Lowercase and tidy text so a quote can be found inside the evidence."""
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split()).strip(" .,;:\"'")


def check_quote(quote, evidence_list):
    """
    Return (quote, found). The quote is cut to under MAX_QUOTE_WORDS words.
    found = True only if the quote really appears in the evidence text.
    """
    words = quote.split()
    if len(words) >= config.MAX_QUOTE_WORDS:
        quote = " ".join(words[: config.MAX_QUOTE_WORDS - 1])
    if not quote.strip():
        return "", False
    all_evidence = _normalise(" ".join(ev["text"] for ev in evidence_list))
    return quote, _normalise(quote) in all_evidence


def _apply_checks(items, model_results):
    """Join the model's answers to our claims (by id) and run the code checks."""
    by_id = {r.claim_id: r for r in model_results}
    checked = []
    for item in items:
        r = by_id.get(item["id"])
        if r is None:
            verdict, reason, quote, better = NOT_ENOUGH, "The judge gave no answer for this claim.", "", ""
        else:
            verdict, reason, quote, better = r.verdict, r.reason, r.evidence_quote, r.better_search_query

        # Guard: with no evidence, the model can only have used its memory.
        if not item["evidence"] and verdict != NOT_ENOUGH:
            reason = "No Wikipedia evidence was found, so this cannot be checked. " + reason
            verdict = NOT_ENOUGH

        quote, quote_found = check_quote(quote, item["evidence"])
        checked.append({
            **item,
            "verdict": verdict,
            "reason": reason.strip(),
            "evidence_quote": quote,
            "quote_found": quote_found,
            "better_search_query": better.strip() if verdict == NOT_ENOUGH else "",
        })
    return checked


def judge_claims(items):
    """
    items: [{"id", "claim", "search_query", "evidence": [ {title,url,text} ]}]
    Returns the same items with verdict, reason, evidence_quote and so on added.
    """
    if not items:
        return []
    result = call_llm(
        step="judge",
        system=prompts.JUDGE_SYSTEM,
        prompt=prompts.build_judge_prompt(items),
        schema=JudgeResult,
        max_output_tokens=2048,
    )
    return _apply_checks(items, result.results)


def reflect_claims(items):
    """
    items: claims to look at again. Each has the new "evidence" list and
    "earlier_reason". Returns re-judged items (one LLM call for all of them).
    """
    if not items:
        return []
    result = call_llm(
        step="reflect",
        system=prompts.REFLECT_SYSTEM,
        prompt=prompts.build_reflect_prompt(items),
        schema=JudgeResult,
        max_output_tokens=1024,
    )
    return _apply_checks(items, result.results)
