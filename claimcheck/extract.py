"""
extract.py - Step A: split an answer into short factual claims (1 LLM call).
"""

from pydantic import BaseModel

import config
from claimcheck import prompts
from claimcheck.llm import call_llm


# The JSON shape we expect back from the model.
class ExtractedClaim(BaseModel):
    claim: str
    search_query: str


class ExtractResult(BaseModel):
    claims: list[ExtractedClaim]


def extract_claims(answer_text):
    """
    Return a list of claims like:
      [{"id": 1, "claim": "...", "search_query": "..."}, ...]
    """
    answer_text = answer_text.strip()[: config.MAX_ANSWER_CHARS]
    result = call_llm(
        step="extract",
        system=prompts.EXTRACT_SYSTEM,
        prompt=prompts.build_extract_prompt(answer_text),
        schema=ExtractResult,
        max_output_tokens=1024,
    )

    claims = []
    for item in result.claims:
        text = item.claim.strip()
        if not text:
            continue  # skip empty claims
        claims.append({
            "id": len(claims) + 1,  # numbering is done in code, not by the model
            "claim": text,
            "search_query": item.search_query.strip() or text,
        })
    return claims[: config.MAX_CLAIMS]  # hard cap, even if the model gives more
