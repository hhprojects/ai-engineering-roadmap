# Paper Reading Agent — Build Plan

> Day-by-day milestone checklist. Tick items off as you ship them. Each milestone ends in a meaningful git commit.

**Total effort target:** ~1–1.5 weekends (roughly 15–20 focused hours) to reach a resume-worthy shippable state.

---

## Pre-flight

- [ ] Create standalone GitHub repo: `github.com/hhprojects/paper-reading-agent`
- [ ] Initialize with `README.md` (the recruiter-facing one — draft per `RESUME_BULLET.md` guidance)
- [ ] Add `.gitignore` (Python) and `LICENSE` (MIT)
- [ ] Set up `uv` or `poetry` for dependency management
- [ ] Create `.env.example` with `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`

---

## Day 1 — Core extraction + chunking pipeline (~5 hours)

**Definition of done:** You can run `python -m src.ingest <arxiv_url>` and see chunked text embedded in pgvector.

- [ ] Spin up PostgreSQL + pgvector locally via Docker Compose (`docker-compose.yml` with postgres:16 + pgvector extension)
- [ ] Write `src/ingest.py`:
  - [ ] arXiv URL → download PDF via `requests`
  - [ ] PDF → raw text via `pymupdf`
  - [ ] Raw text → semantic chunks (target ~500 tokens, 50-token overlap)
  - [ ] Chunks → embeddings via OpenAI `text-embedding-3-small`
  - [ ] Store in pgvector with columns: `paper_id`, `chunk_idx`, `text`, `embedding`, `section_hint`
- [ ] Write a smoke test with 1 real arXiv paper
- [ ] **Commit:** `feat: arxiv ingestion pipeline with pgvector storage`

---

## Day 2 — Agent loop + tool use (~6 hours)

**Definition of done:** Calling `python -m src.agent <arxiv_url>` produces a JSON output with summary + 5 flashcards + 3 quiz questions.

- [ ] Define tool schemas in `src/tools.py`:
  - [ ] `fetch_arxiv_metadata(arxiv_id) -> dict`
  - [ ] `extract_paper_text(arxiv_id) -> str` (wraps ingest)
  - [ ] `semantic_search(query, paper_id, k=5) -> list[chunk]`
  - [ ] `get_section(paper_id, section_name) -> str`
- [ ] Write `src/agent.py` — Claude tool-use loop:
  - [ ] System prompt defining the agent's role (pedagogical paper reader)
  - [ ] Initial user message with paper URL
  - [ ] Loop: send messages → receive tool_use → execute tool → append tool_result → repeat until `end_turn`
  - [ ] Parse final structured output (summary, flashcards, quiz) via Claude's structured output / JSON mode
- [ ] Add basic error handling: retry on tool errors, cap max loop iterations at 15
- [ ] **Commit:** `feat: agent loop with tool use for paper comprehension`

---

## Day 3 — Eval harness + golden set (~4 hours)

**Definition of done:** Running `python -m src.eval` produces a score report on your golden set.

- [ ] Create `evals/golden_set/` with 10 diverse arXiv papers (mix of ML, systems, theory)
- [ ] For each golden paper, manually write:
  - [ ] 1 reference summary (3–5 sentences)
  - [ ] 5 reference flashcards (Q/A pairs)
  - [ ] 3 reference quiz questions with answers
- [ ] Write `src/eval.py`:
  - [ ] Run agent on each golden paper
  - [ ] Score output with LLM-as-judge (Claude grading on rubric: factual accuracy, coverage, question difficulty)
  - [ ] Output `eval_report.json` with per-paper scores + aggregate
- [ ] Design the rubric carefully — document in `evals/RUBRIC.md`
- [ ] **Commit:** `feat: LLM-as-judge eval harness with 10-paper golden set`

---

## Day 4 — FastAPI service + minimal UI (~3 hours)

**Definition of done:** `docker-compose up` gives you a working HTTP API with a minimal web UI.

- [ ] Write `src/api.py`:
  - [ ] `POST /ingest` — accepts arXiv URL or PDF upload
  - [ ] `GET /papers/{id}` — returns status + output
  - [ ] `GET /papers/{id}/flashcards` — returns flashcards only
  - [ ] `POST /papers/{id}/quiz` — generate quiz from stored paper
- [ ] Build a minimal frontend (pick one):
  - [ ] Option A: Streamlit app (fastest)
  - [ ] Option B: Single-file React + Vite page (more resume-appropriate)
- [ ] Update `docker-compose.yml` to include api + postgres
- [ ] **Commit:** `feat: FastAPI service and minimal web UI`

---

## Day 5 — MCP server wrapping (~2 hours)

**Definition of done:** Claude Desktop can successfully call your tools via the MCP server.

- [ ] Install Python MCP SDK: `pip install mcp`
- [ ] Write `src/mcp_server.py`:
  - [ ] Expose 3 tools: `summarize_paper`, `generate_flashcards`, `quiz_me_on_paper`
  - [ ] Each tool calls into the existing agent
- [ ] Add MCP config to your local Claude Desktop (`claude_desktop_config.json`)
- [ ] Record a short demo video/GIF of Claude Desktop calling the tools
- [ ] **Commit:** `feat: expose agent as Model Context Protocol (MCP) server`

---

## Day 6 — Polish + deploy + shipping (~2 hours)

**Definition of done:** Repo has a great README, the app is deployed publicly, and the resume is updated.

- [ ] Rewrite the repo's public README for recruiter skimming (see `RESUME_BULLET.md` for template)
  - [ ] One-line pitch at top
  - [ ] Demo GIF or screenshots
  - [ ] Architecture diagram (ASCII or Mermaid)
  - [ ] "Run locally" section with 3 commands
  - [ ] Eval results section with actual numbers from Day 3
- [ ] Deploy:
  - [ ] Push to Railway or Render (free tier)
  - [ ] Get a public HTTPS URL
  - [ ] Add deployed URL to repo README and resume
- [ ] Update resume:
  - [ ] Replace MyFitLife entry with Paper Reading Agent
  - [ ] Fill in real metrics from Day 3 eval results
  - [ ] See `RESUME_BULLET.md` for exact placement and wording
- [ ] **Commit:** `docs: production-ready README with eval results and deployment link`

---

## Stretch goals (optional, post-ship)

Only consider these if you have time and the core version is genuinely shipped:

- [ ] Support for non-arXiv PDFs (general academic papers)
- [ ] Multi-paper comparison mode ("summarize these 3 papers and find disagreements")
- [ ] Export to Anki deck format
- [ ] Support for video lectures via transcript extraction
- [ ] Swap OpenAI embeddings → local BGE model (shows cost-conscious engineering)

---

## Progress tracker

| Day | Status | Date completed | Commit hash | Notes |
|---|---|---|---|---|
| Pre-flight | ☐ | — | — | — |
| Day 1 — Ingestion | ☐ | — | — | — |
| Day 2 — Agent loop | ☐ | — | — | — |
| Day 3 — Eval | ☐ | — | — | — |
| Day 4 — API + UI | ☐ | — | — | — |
| Day 5 — MCP | ☐ | — | — | — |
| Day 6 — Ship | ☐ | — | — | — |

Update this table as you ship each milestone — it doubles as your dev journal for interview storytelling.
