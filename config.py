"""
config.py - the ONE place for settings.

Everything here can be changed in the .env file without touching code.
Model names and free-tier limits change often, so they are never hardcoded
anywhere else in the project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Folder that contains this file = the project root.
PROJECT_ROOT = Path(__file__).resolve().parent

# Read the .env file (if it exists) into environment variables.
load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name, default):
    """Read a whole number from the environment, or use the default."""
    value = os.getenv(name, "").strip()
    return int(value) if value else default


# --- API keys (never print these) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()  # optional fallback

# --- Mock mode: 1 = canned answers, no network calls to the LLM ---
MOCK_LLM = os.getenv("MOCK_LLM", "1").strip() == "1"

# --- Models (checked against the live model list in Phase 2) ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip() or "gemini-2.5-flash-lite"
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
TEMPERATURE = 0  # 0 = least random, same input gives (almost) same output

# --- Quota protection ---
SESSION_CALL_CAP = _int_env("SESSION_CALL_CAP", 60)   # live calls per app run
DAILY_CALL_CAP = _int_env("DAILY_CALL_CAP", 200)      # live calls per day
WARNING_RATIO = 0.8                                   # warn at 80% of a cap
REQUESTS_PER_MINUTE = _int_env("REQUESTS_PER_MINUTE", 8)
MAX_RETRIES = 2                       # retries after the first try
RETRY_WAITS_SECONDS = [5, 15]         # wait before retry 1, retry 2

# --- Pipeline limits (keep prompts small) ---
MAX_CLAIMS = 8              # claims taken from one answer
MAX_ANSWER_CHARS = 4000     # longer answers are cut to this size
EVIDENCE_CHARS = 1000       # Wikipedia text kept per claim
MAX_REFLECT_CLAIMS = 3      # claims re-searched in the reflect step
MAX_QUOTE_WORDS = 25        # proof quotes must be shorter than this

# --- Folders ---
CACHE_DIR = PROJECT_ROOT / "cache"            # kept in git so the demo works offline
STATE_DIR = PROJECT_ROOT / ".state"           # daily usage counter (not in git)
MOCK_DIR = PROJECT_ROOT / "mock_fixtures"     # canned answers for mock mode
