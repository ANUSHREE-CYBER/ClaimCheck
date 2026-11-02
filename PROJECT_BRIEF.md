# ClaimCheck: Project Brief

**Read this whole file first. It is the single source of truth for this project.**
If anything in a chat message conflicts with this file, the chat message wins. Then update this file so it stays true.

Owner: Anushree (AK). Course: Prompt Engineering (IIS4143), final-year B.Tech CSE.
Due: **Tuesday 3 Nov 2026**. The teacher will look at the project and take a **viva** (oral questions) on it.

---

## 0. The one-paragraph version

ClaimCheck is a small agentic AI tool. You paste an AI-written answer. It splits the answer into individual factual claims, looks up evidence for each claim on Wikipedia, and judges each claim as **Supported**, **Contradicted** or **Not enough evidence**, with a short reason and proof. A Streamlit page shows the answer with each claim colour-coded. It runs on the **free Gemini API**, so protecting the API quota is a core design requirement, not an afterthought.

Guiding principle (say it the same way everywhere): **"Don't trust an AI answer as a whole. Check it claim by claim, and show the proof."**

---

## 1. Why we are building this

- The course needs an **agentic AI project**. It must be efficient, low-cost and eye-catching.
- It must be **demoed live** to the teacher, and AK must **prove it works** and **explain every part** in a viva.
- AI models sometimes state false things confidently (hallucinations). A claim-by-claim checker is a simple, honest way to show prompt engineering plus tool use working together.
- The original working name "Hallucination Hunter" was dropped as cliché. The name is **ClaimCheck**. Use it everywhere.

**AK's situation, which shapes how you build:**
- Very little time (one evening). A working, simple demo beats an impressive broken one.
- AK describes herself as a beginner in AI/ML. She must be able to explain every file. **Write simple, readable, well-commented code. No clever abstractions, no heavy frameworks.** Small files, clear names, one job per function.
- Past experience: a Gemini key ran out very fast on another project because of retries and debugging. **Never burn calls carelessly** (see section 5).

---

## 2. Working rules (read carefully)

### 2.1 Autonomy: work on your own
- Do **not** ask permission for file edits, creating files, running commands, installing free packages, running tests, or making local git commits. Just do them.
- Keep going step by step through the phases in section 7 without waiting. Give a **short summary at the end of each phase** (what was built, what was tested, what is next). Do not narrate every edit.
- You must know **how, when and why** you are building each piece. Before each phase, spend a moment checking it against this brief. Do not guess; if the brief does not cover something, choose the simplest option that fits and note the decision in the Obsidian log.

### 2.2 Hard NEVER rules
1. **Never delete anything on your own.** No `rm`, `del`, `rmdir`, `Remove-Item`, `git clean`, `git reset --hard`, `git push --force`, no overwriting a file in a way that loses content. If something truly must go, move it to a `_to_delete/` folder inside the project and tell AK, or ask.
2. **Never spend money.** No paid APIs, no enabling billing, no paid packages or services, no upgrading plans. If something would cost money, stop and ask.
3. **Never print, log, commit or echo the API key.** Keys live only in `.env` (listed in `.gitignore`). Provide `.env.example` with empty values.
4. **Never claim something works unless you ran it** and saw it work. Report failures plainly, with the real error.
5. **Never add Claude as a co-author or contributor on git commits.** No `Co-authored-by` trailer and no "Generated with Claude Code" line in commit messages.
6. Do not touch files outside the project folder, **except** the one Obsidian note named in section 9.

### 2.3 When you MUST ask AK (only these)
- You need the **Gemini API key** (or a Groq key). Ask AK to put it in `.env` as `GEMINI_API_KEY=...` herself. Do not ask her to paste it into chat. Never print it.
- Anything that would **delete** data, **spend money**, or could not be undone.
- A critical decision that changes the scope or drops a required feature.
- Installing a different Python version on the machine.
- Pushing to GitHub or creating a remote repo. Commit **locally only** unless AK says push.

Everything else: decide and continue.

### 2.4 Quality rules
- **Test everything in the real running app before committing.** Run the actual code, not just read it. For the UI, start Streamlit and test it (use `streamlit.testing.v1.AppTest` for automated UI checks, and confirm the server starts and the page loads). Commit only after it passes.
- Be **brutally honest**. If a result is weak, say so. If a limitation exists, write it down. AK prefers straight feedback to sugarcoating.
- Do not tune prompts to fit the test set and then present the score as general accuracy. Report the real number and say how small the test set is.
- Commit small and often, locally, with clear messages.

### 2.5 Safety for the Claude Code session
In Phase 0, create `.claude/settings.json` that denies destructive commands, as an extra guard on top of the rules above:

```json
{
  "permissions": {
    "deny": [
      "Bash(rm:*)", "Bash(rmdir:*)", "Bash(del:*)", "Bash(Remove-Item:*)",
      "Bash(git clean:*)", "Bash(git reset --hard:*)", "Bash(git push --force:*)"
    ]
  }
}
```
If this syntax is rejected, check the current Claude Code docs and fix it. Do not skip the guard silently; tell AK.

---

## 3. Tech stack

| Part | Choice | Why |
|---|---|---|
| Language | Python. Prefer 3.12 (`py -3.12`), else 3.13, else 3.14 | Most libraries support 3.12. AK's system default is 3.14, which some packages may not support yet. **Test `pip install` of everything in a fresh venv first** and pick the newest version where it all works. |
| Main model | Gemini **Flash-Lite** class model via the `google-genai` SDK | Free tier with the best daily limit. The full "Flash" model has a very small free daily limit (about 20 requests/day), so do **not** use it. |
| Fallback model | Groq (OpenAI-compatible HTTP endpoint, called with `requests`) | Separate free quota. **Optional**, built only after the core works. Only active if `GROQ_API_KEY` is set. |
| Evidence source | Wikipedia (MediaWiki API, no key, plain `requests`) | Free, no key, no LLM calls needed for retrieval. |
| UI | Streamlit | Fast to build, looks good on a projector. |
| Config | `python-dotenv` + a small `config.py` | One place for model names and limits. |
| Validation | `pydantic` | Check that the model's JSON is the right shape. |
| Tests | `pytest` (in mock mode) | Zero API calls while developing. |

**Do not hardcode model names.** Put them in `config.py`/`.env`. Free-tier limits and model names change often. At first live run, list the models available to the key (`client.models.list()`), pick the Flash-Lite variant, and record the choice and the limits you observe in the Obsidian log.

---

## 4. How the tool works (design)

The pipeline is a small **plan -> act -> judge -> reflect** loop. **Maximum 3 live LLM calls per check.**

**Step A. Extract claims (1 LLM call).**
Input: the answer text. Output JSON: up to **8** claims. Each claim is a short, self-contained factual statement (no pronouns like "it" or "he"), plus a `search_query` for Wikipedia. Opinions and vague statements are skipped.

**Step B. Retrieve evidence (0 LLM calls).**
For each claim, call the Wikipedia search API with its `search_query`, take the top result's intro text, and trim to about **1000 characters**. Cache every Wikipedia response on disk. This is the "tool use" part of the agent.

**Step C. Judge (1 LLM call, all claims in one batch).**
Input: all claims with their evidence. Output JSON per claim, in this order: `reason` (short, written first), then `verdict` (`SUPPORTED`, `CONTRADICTED` or `NOT_ENOUGH_EVIDENCE`), `evidence_quote` (a short quote of **under 25 words** copied from the evidence, or empty) and, only for `NOT_ENOUGH_EVIDENCE`, a `better_search_query`.

**Step D. Reflect (0 or 1 LLM call).**
Take the claims marked `NOT_ENOUGH_EVIDENCE` (at most 3). Re-search using their `better_search_query` (0 calls), then re-judge **only those claims** in one more call. **One round only, never a loop without a limit.**

**Output:** per-claim verdicts, counts (supported/contradicted/unverified), an overall "trust" line, and the number of live calls used.

Important honesty points the UI and notes must state:
- A made-up fact with no Wikipedia page comes out as **Not enough evidence**, not as "false". That is correct and honest. Do not turn it into a false "Contradicted".
- It checks **claims against Wikipedia text**, not the model's inner workings. It is weak on very recent events, niche facts, opinions and numbers that differ between sources.
- The AI judge itself can be wrong. That is why the proof quote is shown.

---

## 5. Protect the API quota (core requirement)

Free-tier limits are tight. These rules are part of the design. Build them **before** any real API call is made.

1. **Disk cache.** Every successful LLM response is saved, keyed by a hash of provider + model + temperature + system prompt + prompt + schema. Same input again costs **zero** calls. Never cache errors or invalid JSON. Cache folder: `cache/` (small JSON files; keep it in the repo so the demo works offline from cache).
2. **Mock mode.** `MOCK_LLM=1` makes the LLM wrapper return canned fixtures from `mock_fixtures/` with **no network call**. All development, unit tests and UI work happen in mock mode. Real calls only for deliberate live testing. When mock mode is on, the UI must show a clear **"MOCK MODE"** badge, so mock output is never confused with real results. The final demo must run with `MOCK_LLM=0`.
3. **Call budget with a visible counter.** Count live calls only (cache hits do not count). Defaults: session cap about **60**, daily cap about **200** (stored in `.state/usage.json` with the date). At 80% show a warning. At the cap, **stop** and raise a clear `BudgetExceeded` error that the UI shows as a friendly message. The UI sidebar always shows "Live calls used: X / cap", "Cache hits: Y" and the current model.
4. **Strict retries.** At most **2** retries, only for temporary errors (rate limit or server error), with a growing wait. Each retry counts toward the budget. **Never retry** on bad request, bad key, or invalid output you can fix in code. If the daily quota is exhausted (a 429 saying so), do not retry. Use the fallback if configured, otherwise show a clear message.
5. **Rate-limit throttle.** Keep under the per-minute limit with a simple sliding window (default **8 requests/minute**, configurable). Calls are spaced out automatically. This avoids 429s in the first place.
6. **Keep prompts small.** Trim evidence, cap claims at 8, set `temperature=0`, ask for compact JSON, set a sensible max output size.
7. **Automatic fallback to Groq** when Gemini is rate-limited or exhausted, if a Groq key exists. Optional, built last. Limits are per Google project, so a second key from the same project does not help.

**Pre-run the demo.** Before the demo, run every demo case once in live mode so the answers sit in `cache/`. During the viva these cases load instantly and cannot fail even if the quota is empty. New text typed by the teacher still goes live. If the quota is gone, show the friendly message and fall back to the cached examples.

Expected total real calls for the whole project: roughly 50 to 100. Report the real count at the end.

---

## 6. Prompt engineering concepts to show (this is the course subject)

Keep **all prompts in one file, `prompts.py`**, with a short comment above each naming the technique used. The teacher should be able to read that file and see the course in action.

| Technique (from the syllabus) | Where it appears |
|---|---|
| **C.L.E.A.R. prompting** (Concise, Logical, Explicit, Adaptive, Reflective) | Every prompt is short, ordered, and explicit about the output format. The reflect step is the "Reflective" part. |
| **Zero-shot vs few-shot** | The claim extractor has 2 to 3 few-shot examples. Be ready to explain why (it fixes the output shape and the "self-contained claim" style). |
| **Chain-of-thought** | The judge writes `reason` **before** `verdict`. |
| **Self-consistency** | Optional "strict mode" (off by default): judge twice and compare. It costs more calls, so only add it if time is left. Mention it as an extension either way. |
| **Prompt recipes / libraries** | `prompts.py` is a small prompt library: named, reusable, versioned prompts. |
| **Structured output** | JSON schema validated with pydantic. |

---

## 7. Build plan, in order (time-boxed)

Target times are guides, not promises. Total is about one evening. **If behind schedule, cut in this order:** Groq fallback, then self-consistency, then UI polish, then the "How it works" tab. **Never cut:** caching, mock mode, the call budget, the test run, the demo pre-run, the viva notes.

**Phase 0. Setup (about 15 min)**
- Create the project structure (below), venv, `requirements.txt`, `.gitignore` (include `.env`, `.venv`, `.state/`), `.env.example`, `.claude/settings.json` guard, `git init`.
- Choose the Python version by testing installs. Record it.
- Create/open the Obsidian note and write the status block (section 9).

**Phase 1. Core with mock mode (about 45 min)**
- `llm.py` wrapper: cache, budget, retry, throttle, mock mode, provider switch. `prompts.py`. Extract, retrieve (Wikipedia), judge, reflect, `pipeline.py`.
- Mock fixtures and pytest tests. **All tests pass with zero API calls.**

**Phase 2. First live run (about 30 min)**
- Ask AK for the Gemini key (into `.env`). Check the model list. Run **one** tiny live check and confirm the counter, cache and budget work (run it twice; the second must be a cache hit).

**Phase 3. Streamlit UI (about 30 min)**
- Text box, "Check" button, claim cards colour-coded (green supported / red contradicted / grey not enough evidence), each with reason and evidence quote, summary line, sidebar with calls/cache/model, mock-mode badge, friendly error messages, a button per demo example.
- Optional: a "How it works" tab showing the pipeline and the prompts.

**Phase 4. Evaluation (about 30 min)**
- `eval/` with about **30 labelled claims** (true, false, and unverifiable; only facts you are certain about and that are stable) run through retrieve + judge, plus **5 end-to-end answers**. Report: accuracy, errors caught, false alarms, and the honest limits ("small test set"). Save the results to a file and show them in the UI or README.

**Phase 5. Demo prep (about 20 min)**
- `demo/demo_cases.json` with **5 to 6 cases**: (1) a paragraph with 1 to 2 planted errors, (2) an all-correct paragraph, (3) a made-up person/paper (shows "Not enough evidence"), (4) a tricky case that shows a limitation honestly, (5) one more of AK's choice. Pre-run all in live mode so they are cached. Verify they load with `MOCK_LLM=0` and no budget used.
- Write `README.md` (what it is, how to run, limits).

**Phase 6. Viva notes (about 30 min)**
- Finish the Obsidian note with the viva section (section 9). Final full test, final commit.

**Folder structure (keep it this small):**
```
claimcheck/
  PROJECT_BRIEF.md   README.md   requirements.txt
  .env.example   .gitignore   .claude/settings.json
  config.py
  claimcheck/  llm.py  prompts.py  extract.py  evidence.py  judge.py  pipeline.py
  app.py                      (Streamlit)
  eval/  labelled_claims.json  run_eval.py  results.json
  demo/  demo_cases.json
  mock_fixtures/   cache/   tests/
```

---

## 8. Demo and viva plan

**Demo flow (about 3 minutes):**
1. One sentence: "AI answers can sound right and be wrong. ClaimCheck checks each claim against evidence."
2. Run the planted-error paragraph. Point at the red claim, its reason and the proof quote.
3. Run the made-up-paper case. Explain why it says "Not enough evidence" and not "false".
4. Show the sidebar counter and cache. Explain how the free quota is protected.
5. Show the evaluation numbers, with the honest note about test-set size.
6. Open `prompts.py`. Show the few-shot and chain-of-thought parts.

**If something fails live:** switch to cached demo cases. Say the limit plainly. Do not pretend.

**Be ready to answer (write simple answers in the Obsidian note):**
- What does ClaimCheck do, in two sentences?
- What makes it "agentic"? (It plans queries, uses a tool, judges, and re-searches when unsure.)
- Why split into claims instead of judging the whole answer?
- Why Wikipedia, and what are the limits of that?
- Which prompt-engineering techniques did you use, and where?
- Zero-shot vs few-shot: show it in your own prompts.
- How do you know it works? (Evaluation numbers and their limits.)
- What is a hallucination? Why can a checker be wrong too?
- Why Gemini Flash-Lite? How did you handle the free API limits? (Cache, mock mode, budget, throttle.)
- What would you improve with more time? (More sources, self-consistency, larger test set, web search.)

---

## 9. Obsidian notes (important)

**Note file (the only file outside the project you may touch):**
`C:\Users\Anushree\OneDrive\Documents\OBSIDIAN\WORK\Prompt Eng\Claim Check.md`

- **Read it first. Append to it. Never overwrite or delete existing content.** If it does not exist, create it.
- Language: **the simplest possible**. Short bullet points. No jargon; if a technical word is needed, explain it in plain words the first time. Write as if explaining to a friend. The teacher will read this note.
- Update it at the **end of each phase** (3 to 6 bullets), not after every edit.

**Structure of the note:**

1. **Status block (top, keep updated):** project name, one-line purpose, the guiding principle, stack in plain words, Python version used, milestones done, GitHub repo (none yet), Claude Code session ID if available.
2. **Day log:** a heading per day (`## 3 Nov 2026`), then bullets: what was built, what was tested, what went wrong and how it was fixed, decisions made and why, live calls used so far.
3. **Viva sheet (written in Phase 6):**
   - What the project is, in 5 simple lines
   - How it works, step by step, in plain words (use a tiny example)
   - Where each course concept appears (C.L.E.A.R., few-shot, chain-of-thought, and so on)
   - The API limit problem and how it was handled
   - Evaluation results and their honest limits
   - Limitations
   - Likely questions with short simple answers (from section 8)
   - What I would improve

---

## 10. Things the author of this brief was NOT sure about (verify, do not trust)

- Exact Gemini free-tier limits and current Flash-Lite model names. Verify with the model list and AI Studio.
- The exact `google-genai` call shape for JSON output (`response_mime_type` and a response schema). Check the installed package version and its docs.
- Whether all packages install on Python 3.14.
- Groq free-tier limits and model names (only matters for the optional fallback).
- The `.claude/settings.json` deny-rule syntax.

If any of these differ from this brief, **follow reality, then update this file and note it in the Obsidian log.**

**Verified on 3 Nov 2026 (Phase 2):**
- Model in use: **`gemini-3.5-flash-lite`** (set in `config.py`, override with `GEMINI_MODEL` in `.env`). `gemini-2.5-flash-lite` still appears in the model list but returns 404 "no longer available to new users".
- Free-tier RPM/RPD numbers are not on Google's public docs page any more; they are only visible in AI Studio (aistudio.google.com/rate-limit). Our own caps (8/min, 60/session, 200/day) stay as the safety net.
- JSON output: `response_mime_type="application/json"` + `response_schema=<pydantic class>` works in `google-genai` 2.34.0, and keeps the field order (reason before verdict). Gemini 3.x thinking is set to `MINIMAL` via `thinking_config`.
- The SDK does not retry by itself unless `retry_options` is set, so all retries are ours and all are counted.
- Python 3.12.9; all packages install fine. The `.claude/settings.json` deny syntax `Bash(rm:*)` works (tested: `rm` was blocked).

---

## 11. Definition of done

- [x] `streamlit run app.py` works with `MOCK_LLM=0` on the 5 to 6 demo cases, loaded from cache, with no budget used.
- [x] A fresh live check works, shows the calls counter, and a repeat is a cache hit.
- [x] Tests pass in mock mode. Evaluation ran and the results are saved and honestly reported.
- [x] Budget stop, retry limit and the friendly error message were each tested.
- [x] No key in any file that is committed. No deletions happened. No money spent.
- [x] `README.md` written. Obsidian note complete, including the viva sheet.
- [x] Final summary to AK: what works, what does not, real call count, honest limits.
