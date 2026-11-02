"""
llm.py - the ONLY file that talks to an AI model.

Every LLM call in the project goes through call_llm(). It protects the free
API quota in this order:

  1. Mock mode   -> return a canned answer from mock_fixtures/, no network.
  2. Disk cache  -> same input as before? Return the saved answer, 0 calls.
  3. Budget      -> stop with BudgetExceeded if the session/daily cap is hit.
  4. Throttle    -> wait if we already sent too many requests this minute.
  5. Live call   -> Gemini (main). At most 2 retries, only for temporary errors.
  6. Fallback    -> Groq, only if Gemini is rate-limited/out of quota AND a
                    GROQ_API_KEY exists.

The answer is checked against a pydantic model (the expected JSON shape)
before it is cached or returned.
"""

import hashlib
import json
import time
from collections import deque
from datetime import date

import requests
from pydantic import ValidationError

import config


# ---------------------------------------------------------------------------
# Errors. Each one has a friendly message that the UI can show as-is.
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class for every LLM problem."""


class BudgetExceeded(LLMError):
    """We reached our own call cap. Stop before spending more quota."""


class QuotaExhausted(LLMError):
    """The provider says the daily free quota is used up. Never retried."""


class RateLimited(LLMError):
    """Too many requests per minute. Temporary, so it may be retried."""


class TemporaryError(LLMError):
    """Server error or network problem. Temporary, so it may be retried."""


class BadOutput(LLMError):
    """The model answered, but not in the JSON shape we asked for."""


class ConfigError(LLMError):
    """Something is missing or wrong in .env (for example the API key)."""


# ---------------------------------------------------------------------------
# Counters shown in the UI sidebar. "session" = since the app started.
# ---------------------------------------------------------------------------

STATS = {
    "live_calls": 0,     # real API requests sent (retries included)
    "cache_hits": 0,     # answers served from the disk cache
    "last_provider": None,
    "last_model": None,
}

# Times (in seconds) of recent live calls, for the per-minute throttle.
_recent_call_times = deque()


# ---------------------------------------------------------------------------
# Daily usage file: .state/usage.json = {"date": "2026-11-03", "calls": 12}
# ---------------------------------------------------------------------------

def _usage_file():
    return config.STATE_DIR / "usage.json"


def get_daily_calls():
    """Live calls made today (resets automatically on a new date)."""
    try:
        data = json.loads(_usage_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0
    if data.get("date") != date.today().isoformat():
        return 0
    return int(data.get("calls", 0))


def _add_daily_call():
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"date": date.today().isoformat(), "calls": get_daily_calls() + 1}
    _usage_file().write_text(json.dumps(data), encoding="utf-8")


def budget_status():
    """Numbers for the UI: how much of each cap is used, and a warning flag."""
    session_used = STATS["live_calls"]
    daily_used = get_daily_calls()
    warning = (
        session_used >= config.WARNING_RATIO * config.SESSION_CALL_CAP
        or daily_used >= config.WARNING_RATIO * config.DAILY_CALL_CAP
    )
    return {
        "session_used": session_used,
        "session_cap": config.SESSION_CALL_CAP,
        "daily_used": daily_used,
        "daily_cap": config.DAILY_CALL_CAP,
        "cache_hits": STATS["cache_hits"],
        "warning": warning,
    }


def _check_budget():
    """Raise BudgetExceeded if one more live call would go over a cap."""
    if STATS["live_calls"] >= config.SESSION_CALL_CAP:
        raise BudgetExceeded(
            f"Call budget reached for this session ({config.SESSION_CALL_CAP} live calls). "
            "Cached examples still work. Restart the app to reset the session counter."
        )
    if get_daily_calls() >= config.DAILY_CALL_CAP:
        raise BudgetExceeded(
            f"Daily call budget reached ({config.DAILY_CALL_CAP} live calls today). "
            "Cached examples still work. Try new text again tomorrow."
        )


def _count_live_call():
    """Record one real request (called just BEFORE it is sent)."""
    STATS["live_calls"] += 1
    _add_daily_call()
    _recent_call_times.append(time.monotonic())


# ---------------------------------------------------------------------------
# Throttle: a "sliding window" of the last 60 seconds.
# If we already sent REQUESTS_PER_MINUTE calls in that window, wait.
# ---------------------------------------------------------------------------

def _throttle():
    now = time.monotonic()
    # Forget calls older than 60 seconds.
    while _recent_call_times and now - _recent_call_times[0] >= 60:
        _recent_call_times.popleft()
    if len(_recent_call_times) >= config.REQUESTS_PER_MINUTE:
        wait = 60 - (now - _recent_call_times[0])
        if wait > 0:
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Disk cache: cache/llm/<hash>.json
# ---------------------------------------------------------------------------

def _cache_key(provider, model, system, prompt, schema):
    """A fingerprint of everything that changes the answer."""
    parts = [
        provider,
        model,
        str(config.TEMPERATURE),
        system,
        prompt,
        json.dumps(schema.model_json_schema(), sort_keys=True),
    ]
    return hashlib.sha256("\n||\n".join(parts).encode("utf-8")).hexdigest()


def _cache_path(key):
    return config.CACHE_DIR / "llm" / f"{key}.json"


def _read_cache(key, schema):
    try:
        saved = json.loads(_cache_path(key).read_text(encoding="utf-8"))
        return schema.model_validate(saved["response"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValidationError):
        return None


def _write_cache(key, step, provider, model, result):
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {"step": step, "provider": provider, "model": model,
             "response": result.model_dump()}
    path.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Turning the model's text into a checked Python object
# ---------------------------------------------------------------------------

def parse_json_output(text, schema):
    """Parse the model's text as JSON and check its shape. Raise BadOutput if wrong."""
    text = (text or "").strip()
    # Some models wrap JSON in ```json ... ``` fences. Remove them (a code fix,
    # so no retry is needed).
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return schema.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as error:
        raise BadOutput(
            "The AI answered in an unexpected format, so the result was not used. "
            f"Details: {str(error)[:200]}"
        ) from error


# ---------------------------------------------------------------------------
# Providers. Each returns the raw text, or raises one of the errors above.
# ---------------------------------------------------------------------------

def _call_gemini(system, prompt, schema, max_output_tokens):
    # Imported here so mock mode and tests never need the SDK loaded.
    from google import genai
    from google.genai import errors, types

    # retry_options=None means the SDK itself never retries.
    # We do retries ourselves so every attempt is counted.
    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=60_000, retry_options=None),
    )
    thinking = None
    if config.GEMINI_THINKING_LEVEL:
        thinking = types.ThinkingConfig(thinking_level=config.GEMINI_THINKING_LEVEL)
    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=config.TEMPERATURE,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=thinking,
            ),
        )
    except errors.APIError as error:
        raise _classify_http_error("Gemini", error.code, str(error)) from error
    except Exception as error:  # network problems (timeouts, no internet)
        raise TemporaryError(f"Could not reach Gemini: {type(error).__name__}") from error
    return response.text


def _call_groq(system, prompt, schema, max_output_tokens):
    # Groq speaks the OpenAI chat format, so plain HTTP with requests is enough.
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            json={
                "model": config.GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": config.TEMPERATURE,
                "max_tokens": max_output_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
    except requests.RequestException as error:
        raise TemporaryError(f"Could not reach Groq: {type(error).__name__}") from error
    if response.status_code != 200:
        raise _classify_http_error("Groq", response.status_code, response.text)
    return response.json()["choices"][0]["message"]["content"]


def _classify_http_error(provider, status_code, message):
    """Turn an HTTP error into one of our error types (decides retry or not)."""
    lowered = message.lower()
    if status_code == 429:
        # A daily quota message mentions "per day" (Gemini: "...PerDay...").
        if "perday" in lowered.replace(" ", "") or "per day" in lowered:
            return QuotaExhausted(f"{provider}'s free daily quota is used up.")
        return RateLimited(f"{provider} says too many requests per minute.")
    if status_code is not None and status_code >= 500:
        return TemporaryError(f"{provider} server error ({status_code}).")
    if status_code in (401, 403) or "api key" in lowered or "api_key" in lowered:
        return ConfigError(f"{provider} rejected the API key. Check it in .env.")
    return LLMError(f"{provider} rejected the request ({status_code}): {message[:200]}")


PROVIDERS = {
    "gemini": {"call": _call_gemini, "model": lambda: config.GEMINI_MODEL},
    "groq": {"call": _call_groq, "model": lambda: config.GROQ_MODEL},
}


def _active_providers():
    """Providers that have a key, in order of preference."""
    active = []
    if config.GEMINI_API_KEY:
        active.append("gemini")
    if config.GROQ_API_KEY:
        active.append("groq")
    return active


def _call_with_retries(provider, system, prompt, schema, max_output_tokens):
    """One provider: first try + at most MAX_RETRIES retries for temporary errors."""
    call = PROVIDERS[provider]["call"]
    for attempt in range(config.MAX_RETRIES + 1):
        _check_budget()        # every attempt, retries included, must fit the budget
        _throttle()
        _count_live_call()
        try:
            return call(system, prompt, schema, max_output_tokens)
        except (RateLimited, TemporaryError):
            if attempt == config.MAX_RETRIES:
                raise
            time.sleep(config.RETRY_WAITS_SECONDS[attempt])  # growing wait


# ---------------------------------------------------------------------------
# The one public function
# ---------------------------------------------------------------------------

def _load_mock(step, schema):
    path = config.MOCK_DIR / f"{step}.json"
    return schema.model_validate(json.loads(path.read_text(encoding="utf-8")))


def call_llm(step, system, prompt, schema, max_output_tokens=1024):
    """
    Ask the model and return a validated pydantic object.

    step               short name ("extract", "judge", "reflect"); used for mock files
    system, prompt     the instructions and the input text
    schema             pydantic class describing the JSON we expect back
    """
    # 1. Mock mode: no network at all.
    if config.MOCK_LLM:
        STATS["last_provider"], STATS["last_model"] = "mock", "mock"
        return _load_mock(step, schema)

    providers = _active_providers()
    if not providers:
        raise ConfigError("No API key found. Put GEMINI_API_KEY=... in the .env file.")

    # 2. Cache: check every provider's cache first, so a cached answer is
    #    reused even if it originally came from the fallback provider.
    keys = {}
    for provider in providers:
        model = PROVIDERS[provider]["model"]()
        keys[provider] = _cache_key(provider, model, system, prompt, schema)
        cached = _read_cache(keys[provider], schema)
        if cached is not None:
            STATS["cache_hits"] += 1
            STATS["last_provider"], STATS["last_model"] = provider, model
            return cached

    # 3-6. Live call, falling back to the next provider only on quota problems.
    last_error = None
    for provider in providers:
        model = PROVIDERS[provider]["model"]()
        try:
            text = _call_with_retries(provider, system, prompt, schema, max_output_tokens)
        except (QuotaExhausted, RateLimited) as error:
            last_error = error
            continue  # try the fallback provider, if there is one
        result = parse_json_output(text, schema)  # raises BadOutput; never cached
        _write_cache(keys[provider], step, provider, model, result)
        STATS["last_provider"], STATS["last_model"] = provider, model
        return result

    # Keep the same error type (QuotaExhausted or RateLimited), friendlier text.
    raise type(last_error)(
        f"{last_error} No fallback is available right now. "
        "Cached examples still work; new text needs the quota to reset."
    )
