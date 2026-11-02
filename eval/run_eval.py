"""
run_eval.py - measure how well ClaimCheck works on a small hand-labelled test set.

Run from the project folder:   python eval/run_eval.py

Part 1: 30 labelled claims -> retrieve + judge + reflect (same code as the app).
Part 2: 5 full answers with planted errors -> the whole pipeline (extract too).

Every LLM answer is cached, so running this a second time costs 0 live calls.
Results are saved to eval/results.json (the app's Evaluation tab shows them).
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "import config" works

import config
from claimcheck import llm, pipeline, prompts
from claimcheck.judge import CONTRADICTED, NOT_ENOUGH, SUPPORTED, judge_claims

BATCH_SIZE = config.MAX_CLAIMS  # the app judges at most 8 claims per call, so we do too
LABELS = [SUPPORTED, CONTRADICTED, NOT_ENOUGH]


def check_labelled_claims(claims):
    """Part 1: retrieve + judge + reflect, in batches of 8, like the app does."""
    rows = []
    for start in range(0, len(claims), BATCH_SIZE):
        batch = claims[start: start + BATCH_SIZE]
        notes = []
        items = []
        for i, c in enumerate(batch, start=1):
            items.append({"id": i, "claim": c["claim"], "search_query": c["search_query"],
                          "evidence": pipeline._get_evidence(c["search_query"], notes)})
        judged = judge_claims(items)
        judged, _ = pipeline._reflect(judged, notes)
        for c, j in zip(batch, judged):
            rows.append({
                "claim": c["claim"],
                "expected": c["label"],
                "got": j["verdict"],
                "correct": j["verdict"] == c["label"],
                "reason": j["reason"],
                "evidence_quote": j["evidence_quote"],
                "wikipedia_page": ", ".join(ev["title"] for ev in j["evidence"]) or "(none)",
                "re_checked": j.get("reflected", False),
            })
        for note in notes:
            print("  note:", note)
    return rows


def check_answers(answers):
    """Part 2: the full pipeline on whole answers with planted errors."""
    rows = []
    for a in answers:
        result = pipeline.check_answer(a["text"])
        contradicted = [c["claim"] for c in result["claims"] if c["verdict"] == CONTRADICTED]
        caught = [e for e in a["planted_errors"]
                  if any(e.lower() in claim.lower() for claim in contradicted)]
        false_alarms = [claim for claim in contradicted
                        if not any(e.lower() in claim.lower() for e in a["planted_errors"])]
        rows.append({
            "answer": a["text"],
            "planted_errors": a["planted_errors"],
            "caught": caught,
            "false_alarms": false_alarms,
            "claims_found": len(result["claims"]),
            "verdicts": [f'{c["verdict"]}: {c["claim"]}' for c in result["claims"]],
        })
    return rows


def summarise(claim_rows, answer_rows, live_calls, review_notes=()):
    def count(expected=None, got=None):
        return sum(1 for r in claim_rows
                   if (expected is None or r["expected"] == expected)
                   and (got is None or r["got"] == got))

    total = len(claim_rows)
    correct = sum(r["correct"] for r in claim_rows)
    confusion = {exp: {got: count(exp, got) for got in LABELS} for exp in LABELS}
    planted = sum(len(a["planted_errors"]) for a in answer_rows)
    caught = sum(len(a["caught"]) for a in answer_rows)
    e2e_false_alarms = sum(len(a["false_alarms"]) for a in answer_rows)

    metrics = {
        "accuracy": f"{correct}/{total}",
        "accuracy_percent": round(100 * correct / total, 1),
        "false_claims_caught": f"{count(CONTRADICTED, CONTRADICTED)}/{count(CONTRADICTED)}",
        "true_claims_confirmed": f"{count(SUPPORTED, SUPPORTED)}/{count(SUPPORTED)}",
        "made_up_claims_left_unverified": f"{count(NOT_ENOUGH, NOT_ENOUGH)}/{count(NOT_ENOUGH)}",
        "false_alarms (true claim called false)": count(SUPPORTED, CONTRADICTED),
        "made_up claims given a confident verdict": count(NOT_ENOUGH, SUPPORTED) + count(NOT_ENOUGH, CONTRADICTED),
        "e2e_planted_errors_caught": f"{caught}/{planted}",
        "e2e_false_alarms": e2e_false_alarms,
    }

    lines = [
        f"### Evaluation ({date.today().isoformat()}, model `{config.GEMINI_MODEL}`)",
        "",
        f"**Part 1: {total} hand-labelled claims** (retrieve + judge + reflect)",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Overall accuracy | **{metrics['accuracy']} ({metrics['accuracy_percent']}%)** |",
        f"| False claims caught (marked Contradicted) | {metrics['false_claims_caught']} |",
        f"| True claims confirmed (marked Supported) | {metrics['true_claims_confirmed']} |",
        f"| Made-up claims honestly left as Not enough evidence | {metrics['made_up_claims_left_unverified']} |",
        f"| False alarms (a true claim marked Contradicted) | {metrics['false_alarms (true claim called false)']} |",
        f"| Made-up claims given a confident verdict | {metrics['made_up claims given a confident verdict']} |",
        "",
        "Rows = correct label, columns = what ClaimCheck said:",
        "",
        "| | Supported | Contradicted | Not enough evidence |",
        "|---|---|---|---|",
    ]
    for exp in LABELS:
        lines.append(f"| **{exp}** | " + " | ".join(str(confusion[exp][g]) for g in LABELS) + " |")
    lines += [
        "",
        f"**Part 2: {len(answer_rows)} whole answers** (full pipeline, including claim extraction)",
        "",
        f"- Planted errors caught: **{metrics['e2e_planted_errors_caught']}**",
        f"- False alarms (a claim marked Contradicted that was not a planted error): **{e2e_false_alarms}**",
        "",
        f"Live AI calls for this run: {live_calls} (a re-run is served from the cache).",
        "",
        "**Honest limits:** this is a very small test set (30 claims, 5 answers) written by one person, "
        "using well-known, stable facts and hand-written search queries in Part 1. Real AI answers are "
        "messier, so real-world accuracy will likely be lower. The judge and reflect prompts were not "
        "changed after seeing these results; the extract prompt was changed once (v1 to v2, see notes).",
    ]
    if review_notes:
        lines += ["", "**Notes written after looking at the first run:**", ""]
        lines += [f"- {note}" for note in review_notes]
    return metrics, confusion, "\n".join(lines)


def main():
    data = json.loads((Path(__file__).parent / "labelled_claims.json").read_text(encoding="utf-8"))
    if config.MOCK_LLM:
        sys.exit("Evaluation needs real answers: set MOCK_LLM=0 in .env.")

    calls_before = llm.STATS["live_calls"]
    print(f"Part 1: {len(data['claims'])} labelled claims...")
    claim_rows = check_labelled_claims(data["claims"])
    print(f"Part 2: {len(data['answers'])} whole answers...")
    answer_rows = check_answers(data["answers"])
    live_calls = llm.STATS["live_calls"] - calls_before

    metrics, confusion, summary = summarise(claim_rows, answer_rows, live_calls,
                                            data.get("review_notes_written_after_the_first_run", []))
    results = {
        "date": date.today().isoformat(),
        "model": config.GEMINI_MODEL,
        "prompt_versions": prompts.PROMPT_VERSIONS,
        "live_calls_this_run": live_calls,
        "metrics": metrics,
        "confusion": confusion,
        "summary_markdown": summary,
        "claim_rows": claim_rows,
        "answer_rows": answer_rows,
    }
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(summary)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
