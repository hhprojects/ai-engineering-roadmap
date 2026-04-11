# Resume Bullet Draft

> Fill in the metric placeholders after Day 3 (eval harness) finishes. Use the template that best matches your actual eval numbers.

---

## Primary template (use if eval results are strong)

**Project header:**
```
Paper Reading Agent  —  Open-source                          [date range]
AI Engineer (Solo)
```

**Bullets:**

> - Built an autonomous research-paper agent that plans, extracts, and generates study material (summaries, flashcards, quiz questions) from arXiv papers using a hand-rolled Claude/OpenAI tool-use loop with [N] custom tools.

> - Implemented semantic chunking and RAG retrieval over an ingested paper corpus with PostgreSQL + pgvector (HNSW index), supporting sub-[X]00ms similarity search over [Y]+ chunks.

> - Designed an LLM-as-judge evaluation harness over a [Z]-paper golden set, scoring outputs on factual accuracy, coverage, flashcard quality, and quiz calibration — achieving [A]% average accuracy and [B]% coverage vs manual baselines.

> - Exposed the agent as a Model Context Protocol (MCP) server, enabling Claude Desktop to call `summarize_paper`, `generate_flashcards`, and `quiz_me` as native tools; deployed via FastAPI + Docker on [Railway/Render].

---

## Fallback template (use if eval numbers are weak or qualitative)

> - Built an autonomous research-paper agent that plans, extracts, and generates summaries, flashcards, and quiz questions from arXiv papers using a hand-rolled Claude tool-use loop with 6 custom tools (arXiv metadata, semantic search, section extraction, structured generation).

> - Implemented semantic chunking and RAG retrieval with PostgreSQL + pgvector, storing ~500-token chunks with HNSW indexing for sub-second similarity search.

> - Designed an LLM-as-judge evaluation harness with a 10-paper golden set, scoring outputs on factual accuracy, coverage, and question quality — iterating on prompts and retrieval strategy to measurably improve output consistency.

> - Exposed the agent as a Model Context Protocol (MCP) server for Claude Desktop integration; deployed via FastAPI + Docker Compose with a minimal React UI.

---

## Metric placeholders — what to fill in

| Placeholder | Where to find it | Example |
|---|---|---|
| `[N]` custom tools | Count of tools in `src/tools.py` | 6 |
| `[X]00ms` similarity search | `EXPLAIN ANALYZE` on a vector query | sub-500ms |
| `[Y]+` chunks | Count rows in `chunks` table after ingesting golden set | 1,200+ |
| `[Z]` papers in golden set | Count of golden set papers | 10 |
| `[A]%` flashcard accuracy | Aggregate accuracy score from `eval_report.json` | 87% |
| `[B]%` coverage vs manual | Coverage score from eval | 82% |
| Date range | First commit → ship date | e.g., "May 2026" |

---

## Where to insert on the resume

**Replace:** the entire MyFitLife Fitness Application entry
**Keep:** KakiAlert Crisis Reporting Application (BrainHack Finalist credit is valuable)

**New Projects section order:**

1. **Paper Reading Agent — Open-source** (this project, new)
2. **KakiAlert Crisis Reporting Application** (existing, unchanged)

---

## Linking

Add the repo URL directly under the project header, alongside the date:

```
Paper Reading Agent  —  github.com/hhprojects/paper-reading-agent     [date]
```

If the deployed demo URL is short and memorable, link that too:

```
Paper Reading Agent  —  github.com/hhprojects/paper-reading-agent
                        paper-agent.yourdomain.com                    [date]
```

---

## Interview talking points (bonus)

Save these for when a recruiter asks "tell me about this project":

- **Why a hand-rolled agent loop?** "I wanted to understand the primitives before reaching for a framework. I can explain every line of the tool-use loop, including retry logic and context management."
- **Why pgvector?** "I wanted to avoid introducing a new dependency. The team already uses Postgres, and pgvector's HNSW index gave me sub-second retrieval on 10k+ chunks without an external vector DB."
- **Why LLM-as-judge?** "String metrics like BLEU fail on open-ended generative tasks. An LLM judge lets me score on the dimensions I actually care about — factual accuracy, coverage, and question quality — and it's the same eval approach I used at HTX."
- **What did the eval catch?** "Early versions over-used the abstract and under-used the methods section. I caught this by inspecting per-section coverage scores and fixed it by biasing the agent's initial `semantic_search` queries."
- **What would you do next?** "Two things: swap the OpenAI embeddings for a local BGE model to eliminate cost, and add multi-paper comparison mode so a student can ask 'what do these three papers disagree on?'"
