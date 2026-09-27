# Agentic RAG Router — compound questions, grounded citations, and an access-safe semantic cache

A retrieval-augmented question-answering agent that decides **where** each part of a question should be answered
from (a 10-K filings database, an OpenAI documentation database, or live web search), answers every part from its own
source, and combines the parts into **one answer whose citations stay correct**. On top of that, a **semantic cache
that respects role-based access control (RBAC)**: repeat questions come back fast, and cached answers never cross a
permission boundary.

Everything runs in one notebook, [`agentic_router.ipynb`](agentic_router.ipynb), saved with its outputs.

---

## The problem

A router that picks **one** source per question breaks on compound questions. Asked
*"What was Uber's 2021 revenue and what are the newest LLMs?"*, the original pipeline sent the whole question to the
10-K database. The revenue part was answered and cited. For the LLM part the model found nothing in the filings and
**answered from memory**, uncited and possibly outdated, and it looked just as confident as the cited part.

## What I built

```
 question ──► split into sub-questions ──► for each one, in parallel:
                (defensive JSON parsing)       route ─► 10-K DB  (Qdrant, metadata-filtered)
                                                      ├► OpenAI docs (Qdrant)
                                                      └► web search (SerpApi)
                                               answer + its sources, or "not found"
              ──► renumber citations globally ──► synthesise one answer ──► verify citations in code
                                                                             (fallback if broken)
```

**Access control + cache**

```
 user ──► known? ──► route ──► role may use this source? ──► time-sensitive? ──► live answer, never cached
            │ no                  │ no                            │ no
            ▼                     ▼                               ▼
          DENIED               DENIED                   cache partition for THIS source
       (nothing runs)   (cache never consulted)          HIT ─► stored answer
                                                         MISS ─► pipeline ─► store only if it succeeded
```

## Engineering decisions

| Decision | Why |
|---|---|
| **Split output is parsed defensively**, with fallback to "one question" | The model returns text, sometimes fenced or wrapped in prose. A bad split must never crash the agent. Tested on 12 malformed shapes. |
| **Citations are renumbered in code**, not by the model | Every sub-answer numbers its sources from `[1]`; combined, two documents would both be `[1]`. Brackets inside code (`choices[0]`) are left alone. |
| **The composed answer is verified** | Invented citation numbers, or a part whose citations all vanished, reject the synthesis in favour of a plain layout that is correct by construction. The source list is always built by code. |
| **Grounded prompt with a `NOT_FOUND` marker** | "I don't know" has to be a valid answer, in a form code can check, so a missing fact becomes `ok=False` instead of a guess. |
| **Qdrant metadata filter** for company + fiscal year | Search only sees the filing that matches the question when one exists. |
| **Cache partitioned by source**, not by role | You can only reach a partition after passing the permission check for that source, so the leak is impossible to write. Roles that share a source share its answers, and role changes need no cache purge. |
| **Permission re-checked inside the cache** | Defence in depth: calling the cache directly for a forbidden partition raises `PermissionError`. |
| **Key-term guard** on cache hits | Measured: *"Uber revenue 2020"* is as close to *"…2021"* as a real paraphrase (0.084 vs 0.09). No threshold separates them, so hits also require identical years and company names. |
| **Never cache** denials, time-sensitive questions, or failures | A cached stock price is a wrong answer served fast, and a cached error keeps failing for everyone who asks later. |
| **Audit record for every request** | User, role, route, decision, cache outcome, latency: the table an auditor asks for. |

## Results (saved run)

| | |
|---|---|
| Compound-question checks | 5/5, including answer-level checks (correct year, "not found" for missing data) |
| Single question overhead | exactly +1 LLM call (the split); counted by wrapping the client |
| Parser edge cases | 12/12 offline |
| Concurrency (answer phase) | 6.3 s → 3.8 s, against a 3.7 s floor set by the slowest sub-question |
| Concurrency (end to end) | 9–14 %: split and compose stay sequential (Amdahl's law), so the next win is a faster compose model |
| Cache leak tests | provided self-check 4/4 runs + 7 extra tests |
| Cache hit | ≈ 0.9 s = 829 ms router call + 40 ms lookup; misses 1.6–6 s |

## What reading the output caught (the tests were green)

1. **Wrong year, correct-looking citation.** *"Lyft's 2021 revenue"* came back as $4.095B, which is the **2022** figure.
   The FY2022 filing's revenue table lists 2022/2021/2020 without column headers after chunking, and the model took
   the first column. Fix: a metadata filter on fiscal year, so search returns the FY2021 filing.
2. **An invented answer with real-looking citations.** *"Lyft's 2024 revenue"*: the database has no 2024 filing,
   and the model supplied a number from memory, cited to the 2020 and 2022 filings. Fix: a grounded prompt with a
   `NOT_FOUND` marker; the answer now says the data isn't available.
3. **A 42-second stall** on one web search (measured on the provider's side). The original search call had no
   timeout, so the agent would hang indefinitely. Fix: a 30 s timeout, and the resulting error is never cached.

After (1) and (2), the tests were extended to check the answers themselves, not just the routing.

## Known limitations

- Per-source cache partitions are correct only while permissions cover *whole* sources. Row-level permissions
  (e.g. "Lyft filings only") would need that scope in the partition key.
- The key-term guard knows numbers and a short list of company names; other entities rely on the distance threshold.
- Routing before the cache costs one LLM call on every hit. Looking up across all permitted partitions first would
  give ~40 ms hits, at the cost of the cache rather than the router deciding the source.
- The 10-K collection holds Uber FY2021 and Lyft FY2020–2022, and 414 Lyft chunks have no year label.

## Run it

```bash
conda create -n m3-rag python=3.11 && conda activate m3-rag
pip install -r requirements.txt
python -m ipykernel install --user --name m3-rag
```

Create `.env` next to the notebook (git-ignored):

```
OPENAI_API_KEY=...
SERP_API_KEY=...      # serpapi.com, the free plan is enough
```

Open `agentic_router.ipynb` with the `m3-rag` kernel and run all cells. The first run downloads the
`nomic-embed-text-v1.5` embedding model (~550 MB). A full run takes about 5 minutes, uses about 15 web searches
and costs a few cents in OpenAI calls.

## Stack

Python 3.11 · OpenAI (`gpt-5.6-luna`) for routing, splitting and generation · Qdrant (local) for vector search ·
`nomic-embed-text-v1.5` embeddings · FAISS for the semantic cache · SerpApi for web search · asyncio for concurrency

## Layout

```
agentic_router.ipynb       the whole system, run end to end with outputs
rag_helpers.py             semantic-cache reference implementation (time-sensitivity keyword list is reused)
Agentic_RAG/qdrant_data/   prebuilt vector collections: 10-K filings, OpenAI agents documentation
requirements.txt           pinned versions (transformers 4.48.0 matters for the Nomic model)
```

---

*Built on the starter notebook, helper module and prebuilt vector data from the
[Agent Engineering Bootcamp](https://github.com/hamzafarooq/multi-agent-course) (Module 3, Production Agentic RAG).
The sub-query pipeline, grounding fixes, concurrency, RBAC-aware cache, tests and audit log are my additions.*
