# Architecture

> System design for the paper-reading agent. Reference this when you get stuck on "what goes where."

---

## Component Diagram

```
                              ┌──────────────────┐
                              │  Claude Desktop  │
                              │  (MCP Client)    │
                              └────────┬─────────┘
                                       │ MCP stdio
                                       ▼
 ┌──────────────┐   HTTP    ┌──────────────────────────┐
 │  Web UI      │──────────▶│  FastAPI Service         │
 │  (React /    │◀──────────│  + MCP Server Wrapper    │
 │   Streamlit) │           └──────────┬───────────────┘
 └──────────────┘                      │
                                       │ Python call
                                       ▼
                            ┌──────────────────────┐
                            │  Agent Loop          │
                            │  (Claude tool-use)   │
                            └──────────┬───────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  │                    │                    │
                  ▼                    ▼                    ▼
          ┌─────────────┐      ┌──────────────┐     ┌───────────────┐
          │ Ingestion   │      │ Tools        │     │ Output Parser │
          │ Pipeline    │      │ Registry     │     │ + Validator   │
          └──────┬──────┘      └──────┬───────┘     └───────────────┘
                 │                    │
                 ▼                    ▼
         ┌──────────────┐    ┌────────────────┐
         │  PostgreSQL  │    │  OpenAI /      │
         │  + pgvector  │    │  Anthropic API │
         └──────────────┘    └────────────────┘
```

---

## Data Flow

1. **Input:** arXiv URL (e.g., `https://arxiv.org/abs/2301.12345`) or uploaded PDF
2. **Ingestion** (synchronous, ~10–30s):
   - Download PDF
   - Extract text with pymupdf
   - Detect section boundaries (Abstract, Introduction, Methods, Results, Conclusion)
   - Chunk by sections, then by ~500-token windows with 50-token overlap
   - Embed each chunk with OpenAI `text-embedding-3-small`
   - Store in `chunks` table with metadata
3. **Agent loop** (asynchronous, ~20–60s):
   - Claude receives system prompt + paper metadata
   - Claude calls `semantic_search` to retrieve Abstract + key claims
   - Claude calls `get_section("Methods")` for deep detail
   - Claude synthesizes summary + flashcards + quiz
   - Returns structured JSON
4. **Eval** (offline, not in hot path):
   - Run agent on golden set
   - Score outputs via LLM-as-judge rubric
   - Emit report
5. **Output:** JSON + rendered UI

---

## Database Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE papers (
  id          SERIAL PRIMARY KEY,
  arxiv_id    TEXT UNIQUE,
  title       TEXT NOT NULL,
  authors     TEXT[],
  abstract    TEXT,
  pdf_url     TEXT,
  ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
  id           SERIAL PRIMARY KEY,
  paper_id     INT REFERENCES papers(id) ON DELETE CASCADE,
  chunk_idx    INT NOT NULL,
  section_hint TEXT,
  text         TEXT NOT NULL,
  embedding    vector(1536) NOT NULL
);

CREATE INDEX chunks_embedding_idx
  ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE agent_outputs (
  id         SERIAL PRIMARY KEY,
  paper_id   INT REFERENCES papers(id) ON DELETE CASCADE,
  summary    TEXT,
  flashcards JSONB,
  quiz       JSONB,
  model      TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## Tool Registry

These are the tools the agent can call in its loop. Keep them small and single-purpose — the agent, not the tools, does the reasoning.

| Tool | Input | Output | Purpose |
|---|---|---|---|
| `fetch_arxiv_metadata` | `arxiv_id: str` | `{title, authors, abstract, year}` | Grab paper metadata without full download |
| `ingest_paper` | `arxiv_id: str` | `{paper_id: int, n_chunks: int}` | Run full ingestion pipeline |
| `semantic_search` | `query: str, paper_id: int, k: int` | `list[{chunk, score, section}]` | Retrieve relevant chunks |
| `get_section` | `paper_id: int, section_name: str` | `str` | Fetch an entire section by name |
| `generate_flashcards` | `context: str, n: int` | `list[{question, answer}]` | Structured flashcard generation |
| `generate_quiz` | `context: str, n: int, difficulty: str` | `list[{question, options, answer}]` | Multiple-choice quiz generation |

---

## Agent System Prompt (draft)

```
You are a research paper comprehension agent. Your job is to help a student
deeply understand an academic paper by:

1. Reading the paper using the provided tools
2. Producing a concise summary of the key contributions
3. Generating flashcards that test understanding of the core concepts
4. Generating quiz questions that test recall and reasoning

You have access to tools that let you fetch metadata, ingest the full paper
into semantic chunks, search by query, and fetch specific sections.

Workflow:
- Start by fetching metadata and reading the abstract
- Ingest the paper if not already done
- Use semantic_search to find key claims, methods, and results
- Fetch full sections when you need deeper detail
- Synthesize flashcards that cover: (a) definitions, (b) method steps,
  (c) key results, (d) limitations
- Generate quiz questions at mixed difficulty levels

Return a final structured response with: summary, flashcards (5), quiz (3).
Do not make up content that isn't in the paper. When uncertain, prefer
acknowledging uncertainty over fabrication.
```

---

## Eval Rubric (preview)

LLM-as-judge scores each output on a 1–5 scale across four dimensions:

1. **Factual accuracy** — does the output contradict the paper?
2. **Coverage** — does the summary capture the main contributions?
3. **Flashcard quality** — are questions non-trivial and answers correct?
4. **Quiz difficulty calibration** — are questions meaningfully testing understanding?

Final score: weighted average. Reject any output with factual accuracy < 4.

See `evals/RUBRIC.md` (created on Day 3) for the full rubric prompt given to the judge.

---

## Why These Choices

- **Hand-rolled agent loop over LangChain:** Interview signal. You can explain every line, handle edge cases yourself, and debug without framework opacity. LangChain is easy to learn later.
- **pgvector over Chroma/Pinecone:** You already listed pgvector. One fewer new dependency. Also avoids per-query pricing.
- **LLM-as-judge over BLEU/ROUGE:** String-based metrics fail on open-ended generative tasks. LLM-as-judge is the 2026 standard for generative eval and matches your HTX work.
- **MCP server wrapping:** Negligible extra effort, major keyword payoff. MCP is rising fast and validates the skill on your resume.
- **FastAPI + Docker Compose:** Already on your resume. Deploy target (Railway/Render) has free tier + auto-deploy from git.
