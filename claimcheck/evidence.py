"""
evidence.py - Step B: find evidence on Wikipedia (0 LLM calls).

This is the "tool use" part of the agent. For a search query we ask the
free Wikipedia API for the best-matching article and take its intro text.
Every Wikipedia answer is saved in cache/wiki/ so the same search never goes
to the internet twice (and the demo works offline).
"""

import hashlib
import json
import time

import requests

import config

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks every tool to say who it is in the User-Agent header.
HEADERS = {"User-Agent": "ClaimCheck/1.0 (student course project; Python requests)"}


def trim_text(text, max_chars=None):
    """Cut text to about max_chars, ending at a full sentence if possible."""
    max_chars = max_chars or config.EVIDENCE_CHARS
    text = " ".join(text.split())  # tidy whitespace and newlines
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_full_stop = cut.rfind(". ")
    if last_full_stop > max_chars // 2:
        return cut[: last_full_stop + 1]
    return cut + "..."


def _cache_path(query):
    key = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    return config.CACHE_DIR / "wiki" / f"{key}.json"


_last_request_time = [0.0]
MIN_SECONDS_BETWEEN_REQUESTS = 0.5   # be polite: Wikipedia blocks fast bursts
WIKI_RETRY_WAITS = [3, 8]            # Wikipedia is free, so a retry costs no AI quota


def _fetch_from_wikipedia(query):
    """One HTTP request: search + intro text of the top result together.
    Retries up to 2 times if Wikipedia says "too many requests" (429) or has a server error."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",   # search for the query...
        "gsrsearch": query,
        "gsrlimit": 1,           # ...and keep only the top result
        "prop": "extracts",      # then return that page's text
        "exintro": 1,            # intro section only (short)
        "explaintext": 1,        # plain text, no HTML
        "redirects": 1,
    }
    for attempt in range(len(WIKI_RETRY_WAITS) + 1):
        gap = time.monotonic() - _last_request_time[0]
        if gap < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - gap)
        _last_request_time[0] = time.monotonic()
        response = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
        temporary = response.status_code == 429 or response.status_code >= 500
        if temporary and attempt < len(WIKI_RETRY_WAITS):
            time.sleep(WIKI_RETRY_WAITS[attempt])
            continue
        response.raise_for_status()
        return response.json()


def search_wikipedia(query):
    """
    Return {"title", "url", "text"} for the best Wikipedia match,
    or None if nothing was found.
    """
    query = query.strip()
    if not query:
        return None

    path = _cache_path(query)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = _fetch_from_wikipedia(query)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"query": query, "data": data}, ensure_ascii=False),
                        encoding="utf-8")
        data = {"query": query, "data": data}

    pages = data["data"].get("query", {}).get("pages", {})
    if not pages:
        return None
    page = list(pages.values())[0]
    text = page.get("extract", "").strip()
    if not text:
        return None
    title = page["title"]
    return {
        "title": title,
        "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
        "text": trim_text(text),
    }
