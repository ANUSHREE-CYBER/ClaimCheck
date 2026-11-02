"""
pipeline.py - runs the whole check: plan -> act -> judge -> reflect.

  A. extract  (1 LLM call)   split the answer into claims + search queries  (plan)
  B. evidence (0 LLM calls)  search Wikipedia for each claim                (act / tool use)
  C. judge    (1 LLM call)   verdict for every claim, in one batch           (judge)
  D. reflect  (0 or 1 call)  re-search unclear claims once, re-judge them   (reflect)

So one check uses at most 3 live LLM calls (fewer if answers are cached).
"""

import requests

import config
from claimcheck import llm
from claimcheck.evidence import search_wikipedia
from claimcheck.extract import extract_claims
from claimcheck.judge import CONTRADICTED, NOT_ENOUGH, SUPPORTED, judge_claims, reflect_claims


def _get_evidence(query, notes):
    """Wikipedia lookup that never crashes the check. Returns a list (0 or 1 items)."""
    try:
        found = search_wikipedia(query)
    except requests.RequestException as error:
        notes.append(f"Wikipedia could not be reached for '{query}' ({type(error).__name__}).")
        return []
    return [found] if found else []


def _reflect(judged, notes):
    """Step D: one extra round for unclear claims. Returns (judged, info)."""
    candidates = [c for c in judged
                  if c["verdict"] == NOT_ENOUGH and c["better_search_query"]]
    candidates = candidates[: config.MAX_REFLECT_CLAIMS]
    info = {"candidates": len(candidates), "rejudged": 0}

    to_rejudge = []
    for c in candidates:
        new_evidence = _get_evidence(c["better_search_query"], notes)
        old_titles = {ev["title"] for ev in c["evidence"]}
        # Only spend an LLM call if the new search found a DIFFERENT page.
        if new_evidence and new_evidence[0]["title"] not in old_titles:
            to_rejudge.append({
                "id": c["id"],
                "claim": c["claim"],
                "search_query": c["search_query"],
                "evidence": c["evidence"] + new_evidence,
                "earlier_reason": c["reason"],
            })

    if not to_rejudge:
        return judged, info

    try:
        rejudged = reflect_claims(to_rejudge)
    except (llm.BudgetExceeded, llm.QuotaExhausted, llm.RateLimited) as error:
        # Round-one results are still valid, so keep them and say what happened.
        notes.append(f"The re-check step was skipped: {error}")
        return judged, info

    info["rejudged"] = len(rejudged)
    new_by_id = {r["id"]: r for r in rejudged}
    merged = []
    for c in judged:
        if c["id"] in new_by_id:
            r = new_by_id[c["id"]]
            r["reflected"] = True
            r["first_reason"] = r.pop("earlier_reason")
            r["better_search_query"] = ""
            merged.append(r)
        else:
            merged.append(c)
    return merged, info


def trust_line(counts):
    """One plain sentence that sums up the result."""
    total = sum(counts.values())
    if total == 0:
        return "No checkable factual claims were found."
    if counts[CONTRADICTED]:
        return (f"Careful: {counts[CONTRADICTED]} of {total} claims are contradicted by Wikipedia. "
                "Do not trust this answer as a whole.")
    if counts[SUPPORTED] == total:
        return f"All {total} claims are supported by Wikipedia."
    return (f"{counts[SUPPORTED]} of {total} claims are supported; "
            f"{counts[NOT_ENOUGH]} could not be checked. Treat those with caution.")


def check_answer(answer_text):
    """Run the full check and return everything the UI needs, as a dict."""
    calls_before = llm.STATS["live_calls"]
    hits_before = llm.STATS["cache_hits"]
    notes = []

    answer_text = answer_text.strip()
    if len(answer_text) > config.MAX_ANSWER_CHARS:
        notes.append(f"The answer was long, so only the first {config.MAX_ANSWER_CHARS} characters were checked.")

    # A. Plan: which claims to check and what to search for.
    claims = extract_claims(answer_text)

    # B. Act: use the Wikipedia tool for each claim.
    for c in claims:
        c["evidence"] = _get_evidence(c["search_query"], notes)

    # C. Judge all claims in one call.
    judged = judge_claims(claims)
    for c in judged:
        c["reflected"] = False

    # D. Reflect once on the unclear ones.
    judged, reflect_info = _reflect(judged, notes)

    counts = {SUPPORTED: 0, CONTRADICTED: 0, NOT_ENOUGH: 0}
    for c in judged:
        counts[c["verdict"]] += 1

    return {
        "answer": answer_text,
        "claims": judged,
        "counts": counts,
        "trust_line": trust_line(counts),
        "reflect": reflect_info,
        "live_calls": llm.STATS["live_calls"] - calls_before,
        "cache_hits": llm.STATS["cache_hits"] - hits_before,
        "provider": llm.STATS["last_provider"],
        "model": llm.STATS["last_model"],
        "notes": notes,
    }
