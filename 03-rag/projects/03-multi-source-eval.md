# Project 3 — Multi-Source RAG with Full Evaluation

🟠 **Advanced** · builds on lessons 09–13

> A proper production-shaped RAG: multiple document types into one index, contextual retrieval on ingestion, observability via Langfuse, and a full Ragas-style eval harness reporting retrieval AND generation metrics. This is the project that turns "I built a RAG demo" into "I built a RAG system I can defend."

## What you'll build

A RAG system that:

- Ingests **three document types**: PDFs with tables, HTML web pages, and CSVs, into a unified index with proper metadata.
- Applies **contextual chunking** (lesson 09) at ingestion using Claude Haiku 4.5 + prompt caching.
- Stores chunks in **pgvector** with a `tsvector` column for BM25, all in one Postgres database.
- Runs a **hybrid retrieval + rerank** pipeline at query time (dense + BM25 + RRF + Cohere rerank).
- Traces every request with **Langfuse** so you can see exactly what happened on any given query.
- Evaluates with a **50+ question eval set** measuring both retrieval quality (hit@5, MRR) and generation quality (faithfulness, answer relevance, context precision) using an LLM-as-judge pattern.
- Produces a **markdown evaluation report** that compares at least two chunking strategies (contextual vs. naive).
- Has a **CI check** that blocks PRs which drop the hit@5 or faithfulness below a threshold.

This is a real system. If you can finish it and explain the trade-offs, you have the skill level of a production RAG engineer.

## What you'll learn

- Multi-format document parsing: PDFs (including tables), HTML, CSV.
- Contextual chunking with Claude Haiku and prompt caching — including the cost math.
- pgvector end-to-end: schema design, HNSW indexing, `tsvector` for BM25, filterable hybrid retrieval.
- Langfuse tracing and how to use traces to diagnose specific query failures.
- Building an LLM-as-judge eval for faithfulness, context precision, and answer relevance.
- Running evals in CI and using them as a quality gate.
- Writing an evaluation report that communicates trade-offs to non-engineers.

## Prerequisites

- Python 3.11+
- Projects 1 and 2 completed — you are comfortable with chunking, hybrid retrieval, reranking, and eval basics.
- All 13 lessons of this module read. Lesson 09 (contextual retrieval) and lesson 12 (deep eval) are critical.
- **Postgres 15+ with pgvector installed.** Local Docker is fine (`pgvector/pgvector:pg16`), or use Supabase / Neon's free tier.
- API keys for:
  - **Anthropic** (contextual chunking + generation) — start with the $5 free credit, should last the whole project.
  - **Cohere** (reranking) — free tier.
  - **Langfuse Cloud** free tier, or self-hosted via Docker.

## Tech stack

- Python 3.11+, `asyncio`
- `psycopg` (v3, async) for Postgres
- `pgvector` extension
- `pdfplumber` — PDF parsing with table support
- `beautifulsoup4` — HTML parsing
- `pandas` — CSV handling
- `fastembed` or OpenAI `text-embedding-3-small` — embeddings
- `cohere` — reranker
- `anthropic` — contextual chunking + generation
- `langfuse` — tracing
- `ragas` — optional but recommended for metrics
- `pytest` — eval in CI

## Setup

```bash
mkdir multi-source-rag && cd multi-source-rag
python -m venv .venv && source .venv/bin/activate

pip install \
  psycopg[binary] pgvector \
  pdfplumber beautifulsoup4 pandas \
  fastembed openai cohere anthropic \
  langfuse ragas \
  pydantic pytest

docker run -d --name pgvec \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  pgvector/pgvector:pg16

psql postgresql://postgres:postgres@localhost:5432/postgres \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Environment:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
export ANTHROPIC_API_KEY=...
export COHERE_API_KEY=...
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

## Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  document_id TEXT PRIMARY KEY,
  source_path TEXT NOT NULL,
  doc_type TEXT NOT NULL,          -- 'pdf' | 'html' | 'csv'
  content_hash TEXT NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  chunk_text TEXT NOT NULL,              -- raw text for generation
  indexed_text TEXT NOT NULL,            -- contextualised text for retrieval
  embedding vector(1536),                -- or 384 if you use BGE small
  metadata JSONB NOT NULL DEFAULT '{}',
  chunk_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', indexed_text)) STORED
);

CREATE INDEX chunks_embedding_idx
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX chunks_tsv_idx ON chunks USING gin (chunk_tsv);
CREATE INDEX chunks_document_idx ON chunks(document_id);
CREATE INDEX chunks_metadata_idx ON chunks USING gin (metadata);
```

Note `chunk_text` (original) vs. `indexed_text` (contextualised). You embed `indexed_text` and BM25 over it; you send `chunk_text` to the generator.

## Starter scaffold — ingestion

`ingest.py`:

```python
"""Multi-source ingestion with contextual chunking and prompt caching."""

import asyncio
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber
import psycopg
from anthropic import AsyncAnthropic
from bs4 import BeautifulSoup
from langfuse.decorators import observe
from pgvector.psycopg import register_vector_async

from embedder import embed_batch
from splitter import recursive_split

client = AsyncAnthropic()

CONTEXT_PROMPT = """\
Here is the chunk we want to situate within the whole document.
<chunk>
{chunk_text}
</chunk>
Please give a short, succinct context (one or two sentences) situating this
chunk within the overall document for the purposes of improving search retrieval.
Answer with only the succinct context and nothing else.
"""


@observe()
async def generate_context(document_text: str, chunk_text: str) -> str:
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>\n{document_text}\n</document>",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": CONTEXT_PROMPT.format(chunk_text=chunk_text)},
            ],
        }],
    )
    return response.content[0].text.strip()


# ─── Parsers ─────────────────────────────────────────

def parse_pdf(path: Path) -> list[dict]:
    """Returns text chunks and separate table chunks."""
    with pdfplumber.open(path) as pdf:
        pieces = []
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pieces.append({
                    "text": page_text,
                    "metadata": {"doc_type": "pdf", "page": page.page_number},
                })
            for ti, table in enumerate(page.extract_tables() or []):
                header = table[0]
                for row in table[1:]:
                    row_text = ", ".join(
                        f"{h}: {c}" for h, c in zip(header, row) if c
                    )
                    if row_text.strip():
                        pieces.append({
                            "text": row_text,
                            "metadata": {
                                "doc_type": "pdf",
                                "page": page.page_number,
                                "table": ti,
                                "kind": "table_row",
                            },
                        })
        return pieces


def parse_html(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return [{"text": text, "metadata": {"doc_type": "html"}}]


def parse_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    # one chunk per row
    out = []
    for i, row in df.iterrows():
        row_text = ", ".join(f"{col}: {val}" for col, val in row.items())
        out.append({
            "text": row_text,
            "metadata": {"doc_type": "csv", "row": int(i)},
        })
    return out


# ─── Main loop ───────────────────────────────────────

PARSERS = {
    ".pdf": parse_pdf,
    ".html": parse_html,
    ".htm": parse_html,
    ".csv": parse_csv,
}


async def ingest_file(conn, path: Path):
    parser = PARSERS.get(path.suffix.lower())
    if not parser:
        return
    document_id = str(path.resolve())
    pieces = parser(path)
    full_text = "\n\n".join(p["text"] for p in pieces)
    h = hashlib.sha256(full_text.encode()).hexdigest()

    existing = await conn.fetchrow(
        "SELECT content_hash FROM documents WHERE document_id = $1", document_id
    )
    if existing and existing["content_hash"] == h:
        return

    # Chunk text pieces; table rows stay as whole "chunks"
    chunks: list[dict] = []
    for piece in pieces:
        if piece["metadata"].get("kind") == "table_row":
            chunks.append(piece)
        else:
            for c in recursive_split(piece["text"], size=800, overlap=100):
                chunks.append({"text": c, "metadata": piece["metadata"]})

    # Contextual chunking (cache document; loop chunks)
    async def contextualise(chunk):
        context = await generate_context(full_text, chunk["text"])
        indexed = f"{context}\n\n{chunk['text']}"
        return {**chunk, "indexed_text": indexed}

    # Limit concurrency to keep the cache warm
    sem = asyncio.Semaphore(8)
    async def guarded(c):
        async with sem:
            return await contextualise(c)
    chunks = await asyncio.gather(*[guarded(c) for c in chunks])

    # Embed the contextualised text
    embeddings = await embed_batch([c["indexed_text"] for c in chunks])

    async with conn.transaction():
        await conn.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
        await conn.execute(
            """
            INSERT INTO documents (document_id, source_path, doc_type, content_hash)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (document_id) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                ingested_at = now()
            """,
            document_id, str(path), path.suffix.strip(".").lower(), h,
        )
        for i, (c, emb) in enumerate(zip(chunks, embeddings)):
            await conn.execute(
                """
                INSERT INTO chunks
                  (document_id, chunk_index, chunk_text, indexed_text, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                document_id, i, c["text"], c["indexed_text"], emb, c["metadata"],
            )


async def main(root: Path):
    async with await psycopg.AsyncConnection.connect(
        "postgresql://postgres:postgres@localhost:5432/postgres"
    ) as conn:
        await register_vector_async(conn)
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in PARSERS]
        for p in files:
            print(f"ingesting {p}")
            await ingest_file(conn, p)


if __name__ == "__main__":
    import sys
    asyncio.run(main(Path(sys.argv[1])))
```

## Starter scaffold — retrieval + eval

`retrieval.py` mirrors Project 2's hybrid retriever, but stores are async and backed by Postgres. `eval.py` extends Project 2's eval with three new metrics:

```python
"""Faithfulness, context precision, answer relevance — LLM-as-judge."""

from anthropic import Anthropic
from pydantic import BaseModel

client = Anthropic()
JUDGE_MODEL = "claude-sonnet-4-6"  # pin this; do not upgrade mid-experiment


class EvalScores(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float


def _judge_yes_no(prompt: str) -> bool:
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip().upper()
    return text.startswith("YES") or text.startswith("SUPPORTED")


FAITHFULNESS_CLAIMS = """\
Break this answer into atomic factual claims, one per line:
{answer}
"""

FAITHFULNESS_VERIFY = """\
Context:
{context}

Claim: {claim}

Is the claim directly supported by the context?
Respond with exactly one word: SUPPORTED or UNSUPPORTED.
"""


def faithfulness(answer: str, context: str) -> float:
    claims_resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": FAITHFULNESS_CLAIMS.format(answer=answer)}],
    )
    claims_text = "".join(b.text for b in claims_resp.content if b.type == "text")
    claims = [line.strip("-• ").strip() for line in claims_text.splitlines() if line.strip()]
    if not claims:
        return 1.0
    supported = sum(
        int(_judge_yes_no(FAITHFULNESS_VERIFY.format(context=context, claim=c)))
        for c in claims
    )
    return supported / len(claims)


CONTEXT_PRECISION = """\
Question: {question}
Context chunk: {chunk}

Is this chunk directly relevant to answering the question?
Respond with exactly one word: YES or NO.
"""


def context_precision(question: str, chunks: list[str]) -> float:
    if not chunks:
        return 0.0
    relevant = sum(
        int(_judge_yes_no(CONTEXT_PRECISION.format(question=question, chunk=c)))
        for c in chunks
    )
    return relevant / len(chunks)


ANSWER_RELEVANCE = """\
Original question: {question}
Answer: {answer}

Does the answer address the original question?
Respond with exactly one word: YES or NO.
"""


def answer_relevance(question: str, answer: str) -> float:
    return float(_judge_yes_no(ANSWER_RELEVANCE.format(question=question, answer=answer)))
```

In a production eval you would use Ragas directly; this simplified version is good for learning and shows exactly what Ragas is doing under the hood.

## Must-have requirements

- ✅ Ingest at least 3 document types: PDFs (including ones with tables), HTML pages, CSV files.
- ✅ Extract tables from PDFs as separate chunks with `kind: table_row` metadata.
- ✅ Contextual chunking with Claude Haiku 4.5 + prompt caching at ingestion.
- ✅ pgvector-backed index with HNSW + `tsvector` + metadata filters.
- ✅ Hybrid retrieval (dense + BM25 + RRF) + Cohere rerank at query time.
- ✅ 50+ question eval set with labelled expected chunk IDs.
- ✅ Retrieval metrics: hit@5, MRR.
- ✅ Generation metrics: faithfulness, context precision, answer relevance — LLM-as-judge.
- ✅ Langfuse tracing via `@observe()` decorators.
- ✅ Compare at least 2 chunking strategies (e.g., contextual vs. naive 800-token) and report the delta on all metrics.
- ✅ Generate a markdown eval report (`eval/report.md`) with a table of metrics and a short written analysis.
- ✅ A `pytest` eval check that runs a subset on every PR and fails if hit@5 drops below a threshold.

## Stretch goals (pick at least one)

- **Automated question generation.** Feed each document to an LLM that generates 3–5 test questions per document. Use this to grow the eval set beyond 50 questions with minimal manual labour.
- **Citation tracking.** In the generator prompt, require the LLM to cite chunk IDs for every sentence. Parse citations from the output and check that cited chunks were actually retrieved.
- **CI eval gate.** Wire up a GitHub Actions workflow that runs the eval on PR and posts the result as a comment. Block merging on regression.
- **Per-doc-type metrics.** Break down hit@5 and faithfulness by `doc_type`. You will find that CSVs and tables behave differently from prose, and that's a good insight to surface explicitly.

## Rubric

| Area | 1 point | 2 points | 3 points |
|---|---|---|---|
| **Ingestion** | 3 doc types work | + Tables extracted separately | + Contextual chunking runs with cache hit rate > 80% |
| **Retrieval** | Hybrid works end-to-end | + Reranker wired | + Metadata filtering by doc_type |
| **Eval** | hit@5 + MRR reported | + Faithfulness + context precision reported | + 2 chunking strategies compared, with a written analysis |
| **Observability** | Langfuse traces visible | + You pulled one real failure and explained it from the trace | + Per-query scoring wired into traces |
| **Report** | Markdown exists | + Tables and deltas clear | + Non-technical reader could understand your trade-offs |
| **CI** | Eval runs manually | + `pytest` check exists | + Actual PR-blocking threshold |

Target: 15+ / 18. This project is the difference between "I followed a tutorial" and "I can ship RAG systems at work."

## Common pitfalls

- **Ingestion that does not use caching.** Without `cache_control: ephemeral`, contextual chunking costs 20x more. Test by monitoring `cache_creation_input_tokens` vs. `cache_read_input_tokens` in the API response.
- **Treating tables as prose.** If you splat table rows into a paragraph splitter, you destroy the structure. Always handle tables separately.
- **Over-indexing without dedup.** Tables from PDFs often appear on multiple pages (header repetition). Dedup rows by content hash.
- **Eval questions that only test the easy path.** Include questions that require tables, questions that should return "I don't know," questions that span multiple document types. If all your eval questions are pure prose lookups, contextual retrieval will barely show a lift.
- **Using the same model as judge and answerer.** Self-preference bias. Use Sonnet as the judge; generate answers with Haiku. Or swap — but never the same model.
- **Langfuse tracing off by default.** If `LANGFUSE_PUBLIC_KEY` is missing, the decorators are no-ops. Log a warning at startup if tracing is disabled so you do not waste an afternoon wondering why the dashboard is empty.
- **Reading eval numbers without looking at failures.** Aggregate metrics can improve while specific important queries regress. Always print the failure bucket.

## Cost estimate

- **Contextual chunking** with Haiku + caching: about $1–$3 for a ~5000-chunk corpus.
- **Embeddings:** $0 (local `fastembed`) or ~$1 (OpenAI `text-embedding-3-small`).
- **Full eval run** (50 questions × 4 metrics × LLM judge): about $2–$4 per run.
- **10 iterations during development:** ~$20–$40 total.

Cohere rerank free tier handles the reranker calls; Langfuse free tier handles the observations.

## Deliverables

- A repo with `ingest.py`, `retrieval.py`, `eval.py`, `pytest` config, and CI workflow.
- `eval/questions.json` — 50+ labelled queries across doc types.
- `eval/report.md` — the evaluation report with a metrics table, a chunking comparison, and a short written analysis.
- `README.md` with setup, how to run the pipeline, how to run the eval.
- `NOTES.md` — a debrief: what your metrics showed, which chunking strategy won and by how much, one specific failure you found in the Langfuse trace and what you did about it, and what you would build next.
- Optional: a screenshot of the Langfuse dashboard showing a trace of one real query.

## Going further

This project is the full set of skills required to build production RAG. Natural next steps:

- **Swap Postgres for Supabase** and add row-level security so you can serve multi-tenant customer data from the same schema. Now you have a real SaaS-shaped RAG backend.
- **Add an agentic mode** (lesson 10) where hard questions route to a tool-using agent that iterates on retrieval instead of single-shotting.
- **Add GraphRAG local search** for entity-heavy corpora — keep vector RAG as the default and use the graph for entity-lookup queries.
- **Move evaluation into production** — score every user query in real time with faithfulness + answer relevance, alert when the averages drop, and use the flagged queries as the seed for your next eval iteration.

## References

- Anthropic, *Contextual Retrieval*. https://www.anthropic.com/news/contextual-retrieval
- Anthropic, *Prompt caching*. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- pgvector. https://github.com/pgvector/pgvector
- `pdfplumber` — table-aware PDF parsing. https://github.com/jsvine/pdfplumber
- Langfuse docs. https://langfuse.com/docs
- Ragas documentation. https://docs.ragas.io/
- Supabase, *pgvector guides*. https://supabase.com/docs/guides/ai
- Module 3 lessons 09 (contextual retrieval), 11 (structured retrieval), 12 (deep eval), 13 (production).

---

[← Back to RAG module](../README.md)
