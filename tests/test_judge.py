"""Tests for the code checks that run after the judge answers."""

from claimcheck.judge import ClaimVerdict, _apply_checks, check_quote

EVIDENCE = [{"title": "Mount Everest", "url": "u",
             "text": "Mount Everest is Earth's highest mountain above sea level."}]


def verdict(claim_id, v, quote=""):
    return ClaimVerdict(claim_id=claim_id, reason="r", verdict=v,
                        evidence_quote=quote, better_search_query="q")


def test_quote_found_in_evidence():
    assert check_quote("Earth's highest mountain", EVIDENCE) == ("Earth's highest mountain", True)


def test_made_up_quote_is_flagged():
    assert check_quote("the tallest volcano", EVIDENCE)[1] is False


def test_long_quote_is_cut_below_25_words():
    quote, _ = check_quote(" ".join(["word"] * 40), EVIDENCE)
    assert len(quote.split()) == 24


def test_no_evidence_forces_not_enough_evidence():
    items = [{"id": 1, "claim": "c", "evidence": []}]
    result = _apply_checks(items, [verdict(1, "SUPPORTED")])
    assert result[0]["verdict"] == "NOT_ENOUGH_EVIDENCE"


def test_missing_verdict_becomes_not_enough_evidence():
    items = [{"id": 1, "claim": "c", "evidence": EVIDENCE},
             {"id": 2, "claim": "d", "evidence": EVIDENCE}]
    result = _apply_checks(items, [verdict(1, "SUPPORTED", "highest mountain")])
    assert result[0]["verdict"] == "SUPPORTED"
    assert result[1]["verdict"] == "NOT_ENOUGH_EVIDENCE"


def test_better_query_kept_only_for_unclear_claims():
    items = [{"id": 1, "claim": "c", "evidence": EVIDENCE}]
    assert _apply_checks(items, [verdict(1, "CONTRADICTED")])[0]["better_search_query"] == ""
    assert _apply_checks(items, [verdict(1, "NOT_ENOUGH_EVIDENCE")])[0]["better_search_query"] == "q"
