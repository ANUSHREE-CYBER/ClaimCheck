# ClaimCheck

> **Don't trust an AI answer as a whole. Check it claim by claim, and show the proof.**

ClaimCheck is a small agentic AI tool for the Prompt Engineering course (IIS4143).
You paste an answer written by an AI. ClaimCheck:

1. splits it into short, self-contained factual **claims** (1 LLM call),
2. looks up **evidence** for each claim on Wikipedia (0 LLM calls; this is the tool use),
3. **judges** every claim in one batch as **Supported**, **Contradicted** or **Not enough evidence**, with a reason and a proof quote (1 LLM call),
4. **reflects**: unclear claims (max 3) are searched again with a better query and re-judged once (0 or 1 LLM call).

So a check costs **at most 3 live LLM calls**, and 0 if it is already cached.
It runs on the **free Gemini API** (`gemini-3.5-flash-lite`).

## How to run

Needs Python 3.12 (tested with 3.12.9) on Windows.

You need **your own free Gemini API key** (from Google AI Studio, https://aistudio.google.com/apikey; no billing needed).
The key goes only in a `.env` file that you copy from `.env.example`. `.env` is in `.gitignore`, so it is never committed.
Set `MOCK_LLM=0` in `.env` for real checks (the example file starts in mock mode).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env        # then put your key in .env: GEMINI_API_KEY=...
.venv\Scripts\python -m streamlit run app.py
```

- `MOCK_LLM=1` in `.env`: fake, ready-made AI answers (no API calls). The page shows a red **MOCK MODE** badge.
- `MOCK_LLM=0`: real checks. The demo examples are already in `cache/`, so they load instantly and cost 0 calls even without quota.

Other commands:

```powershell
.venv\Scripts\python -m pytest              # 41 tests, mock mode, internet blocked, 0 API calls
.venv\Scripts\python eval\run_eval.py       # evaluation (served from cache after the first run)
```

## Project layout

| File | Job |
|---|---|
| `config.py` | All settings: model names, call caps, limits. Read from `.env`. |
| `claimcheck/llm.py` | The only file that talks to an AI: mock mode, cache, budget, throttle, retries, Groq fallback. |
| `claimcheck/prompts.py` | The prompt library. Every prompt, named and versioned, with the technique it uses. |
| `claimcheck/extract.py` | Step A: answer to claims. |
| `claimcheck/evidence.py` | Step B: Wikipedia search (cached on disk). |
| `claimcheck/judge.py` | Steps C and D: judge and reflect, plus code checks on the AI's output. |
| `claimcheck/pipeline.py` | Runs A, B, C, D in order. |
| `app.py` | The Streamlit page. |
| `eval/` | 30 labelled claims and 5 answers, the eval script, and the results. |
| `demo/demo_cases.json` | The 5 demo cases (pre-run, in the cache). |
| `mock_fixtures/` | Fake AI answers for mock mode and tests. |
| `cache/` | Saved AI and Wikipedia answers (kept in git so the demo works offline). |

## Prompt engineering techniques used

| Technique | Where |
|---|---|
| C.L.E.A.R. | All prompts: short numbered rules, explicit JSON format; reflect = "Reflective". |
| Few-shot | Extract prompt: 3 worked examples (fix the output shape and the self-contained claim style; one keeps a wrong fact on purpose). |
| Zero-shot | Judge prompt: rules only, no examples. |
| Chain-of-thought | Judge writes `reason` **before** `verdict` (enforced by the JSON schema field order too). |
| Structured output | JSON schema passed to Gemini and checked with pydantic. |
| Prompt library / versions | `prompts.py`, with `PROMPT_VERSIONS`. Extract is at v2 (see below). |
| Self-consistency | Not built (would double the calls). Listed as an extension. |

## Protecting the free API quota

- **Disk cache**: same input means 0 calls. Errors and invalid output are never cached.
- **Mock mode** for all development and tests.
- **Call budget**: 60 per app run, 200 per day (saved in `.state/usage.json`), a warning at 80%, then a hard stop with a friendly message.
- **Retries**: at most 2, only for temporary errors (rate limit or server error), with a growing wait (5 s, 15 s). Every retry counts. Never retried: bad request, bad key, bad output, daily quota gone.
- **Throttle**: at most 8 requests per minute (sliding window).
- **Small prompts**: max 8 claims, about 1000 characters of evidence per claim, temperature 0, minimal hidden "thinking".
- **Groq fallback** (optional, used only if `GROQ_API_KEY` is set). The code is unit-tested with a fake; **not tested live** (no Groq key).

## Evaluation (honest)

From `eval/results.json`:

- **30 labelled claims** (retrieve + judge + reflect): **29/30 correct (96.7%)**. True claims confirmed 12/12, false claims caught 11/12, made-up claims left as "Not enough evidence" 6/6, **0 false alarms**.
- **5 whole answers** (full pipeline): planted errors caught **3/4**, 1 "false alarm". That false alarm is really the author's labelling mistake: current Wikipedia says the Great Pyramid was tallest "for more than 3,700 years", not 3,800.
- This is a **very small test set** of famous, stable facts written by one person. Real AI answers are messier, so expect lower real-world accuracy.
- **Bug found and fixed:** extract prompt v1 silently dropped statements the model "knew" were false, so the errors were never checked. v2 fixed it: the demo's planted errors are now all caught. Part 2 scores were the same with v1 and v2. The v1 results are in `eval/results_extract_v1.json`.

## Limitations

- A made-up fact with no Wikipedia page comes out as **Not enough evidence**, not "false". That is correct and honest.
- It checks claims against the **intro of one Wikipedia page** (about 1000 characters). Facts further down the page, myths (for example "the Great Wall is visible from the Moon") and niche details often stay unchecked.
- Weak on recent events, opinions, and **numbers that differ between sources or change over time** (Tokyo's population: one page says 33 million, another 37 million).
- "Missing" is not "wrong": a claim the evidence never mentions is not marked Contradicted, so some errors are missed.
- The AI judge itself can be wrong. That is why every verdict shows its proof quote, and the code checks that the quote really appears in the evidence.
- Wikipedia sometimes rate-limits fast requests. The tool waits and retries; if that still fails, the claim is marked Not enough evidence with a note.

## What I would improve

More evidence sources (full articles, web search), self-consistency (judge twice and compare), a bigger and more varied test set, and highlighting each claim inside the original answer.
