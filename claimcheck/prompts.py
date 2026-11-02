"""
prompts.py - the prompt library for ClaimCheck.

Every prompt the project sends to the AI lives in this one file. Each prompt
has a name and a version, and a comment saying which prompt-engineering
technique it shows. (If a prompt changes, its text changes, so the cache
automatically treats it as a new question.)

All prompts follow C.L.E.A.R.:
  Concise    - short rules, small inputs (trimmed evidence, max 8 claims)
  Logical    - rules are numbered in the order the model should apply them
  Explicit   - the exact JSON output format is spelled out
  Adaptive   - the reflect prompt adapts to what went wrong in round one
  Reflective - the reflect step makes the model look at its first answer again
"""

PROMPT_VERSIONS = {
    "extract": "extract-v1",
    "judge": "judge-v1",
    "reflect": "reflect-v1",
}


# ---------------------------------------------------------------------------
# 1. EXTRACT CLAIMS
# Technique: FEW-SHOT prompting (3 worked examples) + structured JSON output.
# Why few-shot here: the examples fix the output shape and show the
# "self-contained claim" style (no "it"/"she") and that opinions are skipped.
# Zero-shot versions of this prompt tend to merge facts or keep pronouns.
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """You extract checkable factual claims from a text.

Rules:
1. A claim is one short factual statement that could be checked on Wikipedia.
2. Make each claim self-contained: replace pronouns (it, he, she, they, this) with the name they refer to.
3. Skip opinions, advice, predictions and vague statements.
4. Keep the original meaning exactly, including numbers and dates. Never correct a claim, even if you think it is wrong.
5. Return at most 8 claims, the most important first.
6. For each claim, write a search_query: 2 to 6 words naming the Wikipedia article most likely to contain the fact.

Output only JSON in this format:
{"claims": [{"claim": "...", "search_query": "..."}]}
If there are no factual claims, output {"claims": []}."""

EXTRACT_EXAMPLES = """Example 1
Text: Marie Curie was a brilliant scientist. She won the Nobel Prize in Physics in 1903 and later a second Nobel Prize in Chemistry. She was born in Warsaw.
Output: {"claims": [
 {"claim": "Marie Curie won the Nobel Prize in Physics in 1903.", "search_query": "Marie Curie"},
 {"claim": "Marie Curie won a Nobel Prize in Chemistry.", "search_query": "Marie Curie"},
 {"claim": "Marie Curie was born in Warsaw.", "search_query": "Marie Curie"}]}

Example 2
Text: Python is the best language for beginners. It was created by Guido van Rossum and first released in 1991.
Output: {"claims": [
 {"claim": "Python was created by Guido van Rossum.", "search_query": "Python programming language"},
 {"claim": "Python was first released in 1991.", "search_query": "Python programming language"}]}

Example 3
Text: Honestly, I think you should visit soon, the weather is lovely and you will have fun.
Output: {"claims": []}"""


def build_extract_prompt(answer_text):
    return f"""{EXTRACT_EXAMPLES}

Now do the same for this text.
Text: {answer_text}
Output:"""


# ---------------------------------------------------------------------------
# 2. JUDGE CLAIMS (all claims in ONE call, to save quota)
# Technique: ZERO-SHOT prompting with CHAIN-OF-THOUGHT.
# The model must write "reason" BEFORE "verdict", so it thinks first and
# decides second. It may only use the given evidence (grounding), and must
# copy a short proof quote so a human can check the judge too.
# ---------------------------------------------------------------------------

JUDGE_RULES = """For each claim, fill these fields in this order:
1. reason: one or two short sentences comparing the claim with its evidence. Think first, decide second.
2. verdict: one of
   - SUPPORTED: the evidence clearly says the claim is true.
   - CONTRADICTED: the evidence clearly says something that makes the claim false (for example a different date, number, person or place).
   - NOT_ENOUGH_EVIDENCE: the evidence does not mention the fact, or is about something else. Missing information is NOT a contradiction.
3. evidence_quote: copy a short exact phrase (under 25 words) from the evidence that shows your verdict. Use "" for NOT_ENOUGH_EVIDENCE.
4. better_search_query: only for NOT_ENOUGH_EVIDENCE, a better 2 to 6 word Wikipedia search for this fact. Otherwise "".

Output only JSON in this format, one result per claim, in the same order:
{"results": [{"claim_id": 1, "reason": "...", "verdict": "...", "evidence_quote": "...", "better_search_query": "..."}]}"""

JUDGE_SYSTEM = f"""You are a careful fact-checker. Judge each claim using ONLY the evidence given with it. Do not use your own memory.

{JUDGE_RULES}"""


def _format_evidence(evidence_list):
    """Turn a list of evidence dicts into text blocks for the prompt."""
    if not evidence_list:
        return "(no Wikipedia page found)"
    blocks = []
    for ev in evidence_list:
        blocks.append(f'[Wikipedia: {ev["title"]}] {ev["text"]}')
    return "\n".join(blocks)


def build_judge_prompt(items):
    """items: list of dicts with keys id, claim, evidence (list of evidence dicts)."""
    parts = []
    for item in items:
        parts.append(
            f"Claim {item['id']}: {item['claim']}\n"
            f"Evidence {item['id']}: {_format_evidence(item['evidence'])}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# 3. REFLECT (re-judge unclear claims with new evidence, ONE round only)
# Technique: REFLECTION (the "R" in C.L.E.A.R.) + chain-of-thought.
# The model sees its own earlier reason and new evidence from a better
# search, then judges again. Only claims that were NOT_ENOUGH_EVIDENCE come
# here (max 3), so this costs at most one extra call.
# ---------------------------------------------------------------------------

REFLECT_SYSTEM = f"""You are a careful fact-checker taking a second look.
In the first round these claims could not be checked, because the evidence did not mention them.
A second Wikipedia search found more evidence. Read your earlier reason, then judge again using ONLY the evidence given. Do not use your own memory.
If the evidence still does not mention the fact, NOT_ENOUGH_EVIDENCE is the right and honest answer.

{JUDGE_RULES}"""


def build_reflect_prompt(items):
    """items: like build_judge_prompt, plus 'earlier_reason' for each claim."""
    parts = []
    for item in items:
        parts.append(
            f"Claim {item['id']}: {item['claim']}\n"
            f"Your earlier reason: {item['earlier_reason']}\n"
            f"Evidence {item['id']}: {_format_evidence(item['evidence'])}"
        )
    return "\n\n---\n\n".join(parts)
