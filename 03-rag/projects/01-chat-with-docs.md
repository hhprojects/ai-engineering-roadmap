# Project 1 — Chat with Your Docs

🟢 **Beginner** · builds on lessons 01–05

> The "Hello World" of RAG: load your own files, embed them, and answer questions from them. Every later project in this module builds on this baseline, so get it clean.

## What you'll build

A command-line app that:

- Indexes a folder of Markdown / text files into a local SQLite database with embeddings.
- Accepts natural-language questions from a CLI prompt.
- Retrieves the top-k relevant chunks and sends them to an LLM with a grounding prompt.
- Shows the answer plus the source chunks it was built on.
- Handles "I don't know" gracefully when no good source exists.
- Re-indexes incrementally using content hashes.

Under 400 lines of Python. Runs entirely on your laptop with no API key required for the embedding step (uses `fastembed` locally); the LLM step uses any provider you like.

## What you'll learn

- How the full RAG pipeline fits together end-to-end.
- Recursive character text splitting and how chunk size affects retrieval.
- Using SQLite as a vector store for small corpora (yes, really — it works fine).
- Building a grounding prompt that actually produces grounded answers.
- How to add an incremental re-index based on content hashes.
- How to swap local `fastembed` for a hosted embedder without rewriting the pipeline.

## Prerequisites

- Python 3.11+
- You have read lessons 01–05 of this module (or you are willing to read them when you get stuck).
- You have a **free** LLM provider set up. The easiest three:
  - **Groq** (free tier is generous, very fast) — `GROQ_API_KEY`.
  - **Google AI Studio** (Gemini Flash free tier) — `GEMINI_API_KEY`.
  - **Anthropic** (Claude Haiku, $5 free credits on signup) — `ANTHROPIC_API_KEY`.
- Basic comfort with Python, `pip`, and the command line.

## Tech stack

- Python 3.11+
- `fastembed` — local CPU embeddings, no API key needed (uses `BAAI/bge-small-en-v1.5`, 384 dims)
- `sqlite-vec` — sqlite extension for vector search (alternatively: pure `numpy` matrix mult for tiny corpora)
- `typer` — CLI framework
- `pydantic` — typed data models
- `anthropic` / `groq` / `openai` — whichever LLM provider you pick
- `python-frontmatter` — pulls title/metadata from Markdown files

## Setup

```bash
mkdir chat-with-docs && cd chat-with-docs
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install \
  fastembed \
  sqlite-vec \
  typer \
  pydantic \
  python-frontmatter \
  anthropic
```

Export your provider key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or GROQ_API_KEY=..., or GEMINI_API_KEY=...
```

Create a folder of source documents to play with. Clone a small open-source project's docs or use your own personal notes — 20–100 Markdown files is ideal.

## Starter scaffold

Create a single file `chat.py`:

```python
"""
chat.py — a minimal RAG CLI.

Usage:
  python chat.py index ./docs
  python chat.py ask "what is hybrid search?"
  python chat.py ask "what does the readme say about installation?" --k 8
"""

import hashlib
import json
import sqlite3
import struct
from pathlib import Path
from typing import Optional

import frontmatter
import sqlite_vec
import typer
from anthropic import Anthropic
from fastembed import TextEmbedding
from pydantic import BaseModel

app = typer.Typer(add_completion=False)

DB_PATH = Path("chat.sqlite")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SYSTEM_PROMPT = """\
You answer questions strictly from the provided context snippets.
Rules:
- If the context does not contain the answer, say "I don't know based on the provided documents."
- Cite the chunk IDs you used in brackets, like [3] [7].
- Be concise. Prefer quoting or paraphrasing the context over adding your own knowledge.
"""


class Chunk(BaseModel):
    id: int
    document_id: str
    chunk_index: int
    text: str
    source_path: str


# ─────────────────────────────────────────── DB setup ────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
          document_id TEXT PRIMARY KEY,
          source_path TEXT NOT NULL,
          content_hash TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
          chunk_index INTEGER NOT NULL,
          text TEXT NOT NULL,
          source_path TEXT NOT NULL
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors
        USING vec0(
          embedding float[{EMBED_DIM}]
        )
    """)
    return conn


# ─────────────────────────────────────────── Chunking ────

def recursive_split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """A simple recursive splitter.  Tries paragraphs → sentences → words."""
    separators = ["\n\n", "\n", ". ", " "]

    def _split(t: str, sep_idx: int) -> list[str]:
        if len(t) <= size:
            return [t] if t.strip() else []
        if sep_idx >= len(separators):
            # hard wrap
            return [t[i:i + size] for i in range(0, len(t), size - overlap)]
        sep = separators[sep_idx]
        parts = t.split(sep)
        chunks, current = [], ""
        for p in parts:
            piece = (p + sep) if sep != " " else (p + " ")
            if len(current) + len(piece) > size and current:
                chunks.append(current.strip())
                current = current[-overlap:] if overlap else ""
            current += piece
        if current.strip():
            chunks.append(current.strip())
        # If any chunk is still too large, recurse with the next separator
        out = []
        for c in chunks:
            out.extend(_split(c, sep_idx + 1) if len(c) > size else [c])
        return out

    return _split(text, 0)


# ─────────────────────────────────────────── Indexing ────

def vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def load_document(path: Path) -> tuple[str, str]:
    """Returns (document_id, text)."""
    fm = frontmatter.load(path)
    text = fm.content or path.read_text(encoding="utf-8", errors="ignore")
    document_id = str(path.resolve())
    return document_id, text


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@app.command()
def index(folder: Path):
    """Walk FOLDER, chunk and embed every .md/.txt file, write to the index."""
    conn = get_db()
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix in {".md", ".txt"}]
    typer.echo(f"Found {len(files)} files under {folder}")

    existing = {row[0]: row[1] for row in conn.execute("SELECT document_id, content_hash FROM documents")}
    seen = set()

    for path in files:
        document_id, text = load_document(path)
        seen.add(document_id)
        h = content_hash(text)
        if existing.get(document_id) == h:
            continue  # unchanged
        # Delete previous chunks for this doc
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        chunks = recursive_split(text)
        if not chunks:
            continue
        embeddings = list(embedder.embed(chunks))
        for ix, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cursor = conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, text, source_path) "
                "VALUES (?, ?, ?, ?)",
                (document_id, ix, chunk, str(path)),
            )
            chunk_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)",
                (chunk_id, vec_to_blob(emb.tolist())),
            )
        conn.execute(
            "INSERT INTO documents (document_id, source_path, content_hash) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(document_id) DO UPDATE SET content_hash = excluded.content_hash",
            (document_id, str(path), h),
        )
        typer.echo(f"  indexed: {path.name}  ({len(chunks)} chunks)")

    # Drop documents that no longer exist
    gone = set(existing) - seen
    for document_id in gone:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        typer.echo(f"  removed: {document_id}")

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    typer.echo(f"Index has {total} chunks.")


# ─────────────────────────────────────────── Querying ────

def retrieve(conn, query: str, k: int, embedder: TextEmbedding) -> list[Chunk]:
    query_emb = next(iter(embedder.embed([query])))
    rows = conn.execute(
        f"""
        SELECT chunks.id, chunks.document_id, chunks.chunk_index, chunks.text, chunks.source_path
        FROM chunk_vectors
        JOIN chunks ON chunks.id = chunk_vectors.rowid
        WHERE chunk_vectors.embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (vec_to_blob(query_emb.tolist()), k),
    ).fetchall()
    return [
        Chunk(id=r[0], document_id=r[1], chunk_index=r[2], text=r[3], source_path=r[4])
        for r in rows
    ]


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {Path(c.source_path).name})\n{c.text}"
        for i, c in enumerate(chunks)
    )


@app.command()
def ask(question: str, k: int = 5):
    """Ask a question against the indexed corpus."""
    conn = get_db()
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    chunks = retrieve(conn, question, k, embedder)
    if not chunks:
        typer.echo("No documents in the index — run `index` first.")
        raise typer.Exit(1)

    context = format_context(chunks)
    client = Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    answer = "".join(block.text for block in response.content if block.type == "text")

    typer.echo("\n─── Answer ───")
    typer.echo(answer)
    typer.echo("\n─── Sources ───")
    for i, c in enumerate(chunks, start=1):
        typer.echo(f"  [{i}] {Path(c.source_path).name}  (chunk {c.chunk_index})")


if __name__ == "__main__":
    app()
```

That is a complete, working RAG CLI. Run it:

```bash
python chat.py index ./docs
python chat.py ask "what does the project README say about testing?"
```

## Must-have requirements

- ✅ Indexes Markdown and plain text files from a folder.
- ✅ Uses recursive character splitting (NOT fixed-size). 800 tokens / 100 overlap defaults.
- ✅ Stores embeddings in `sqlite-vec` (or pgvector, or Chroma — pick one; the example uses sqlite-vec).
- ✅ Accepts CLI queries and prints the answer.
- ✅ Shows retrieved chunks with source file names and chunk indices.
- ✅ Refuses to answer when no good context is found ("I don't know" path, tested).
- ✅ Incremental re-indexing that skips unchanged files via content hashing.
- ✅ Persists the index between runs.
- ✅ Uses `fastembed` or any provider embedder, selectable via config.

## Stretch goals (pick at least one)

- **Multi-turn conversation.** Maintain a rolling history of the last 4 turns and resolve pronouns using the query rewriter from lesson 08.
- **Chunking comparison.** Add a `compare` command that runs the same question under two different chunking strategies (fixed-size vs. recursive) and shows the retrieved chunks side-by-side. Use this to build intuition for why chunking matters.
- **Metadata filtering.** Add a `--source` flag that only searches files matching a glob pattern, e.g., `python chat.py ask "..." --source "docs/api/*.md"`.
- **Small-to-big retrieval.** Embed sentence-sized chunks but return the surrounding paragraph (2–3 chunks before/after) as the context sent to the LLM.

## Rubric

Score yourself before declaring the project done.

| Area | 1 point | 2 points | 3 points |
|---|---|---|---|
| **Indexing** | Indexes files end-to-end | + Incremental via hashing | + Handles deletions and renames cleanly |
| **Retrieval** | Works for basic questions | + Grounded prompt refuses to answer when no context | + At least one stretch goal implemented |
| **CLI UX** | Runs without crashes | + Clear source display | + `--help`, meaningful errors, configurable `k` |
| **Code quality** | Works | + Pydantic models for chunks | + Separation of indexing / retrieval / generation; unit tests |
| **Eval** | You can name your failure modes | + You built 10 test questions | + You ran hit@5 and wrote down the number |

Total 11+ / 15 is a strong project worth showing in a portfolio.

## Common pitfalls

- **Chunks that are too small.** 200-token chunks look fine until the answer needs context from the surrounding paragraph. Start at 800.
- **Chunks that are too large.** 2000-token chunks retrieve coarsely and the LLM wastes context. 800 is the default for a reason.
- **No grounding prompt.** Without "answer only from this context," the LLM will helpfully hallucinate. Always include the refusal clause.
- **Forgetting to normalise sources.** File paths that differ by absolute vs. relative form create duplicate entries. Use `Path.resolve()` consistently.
- **Mixing chunking formats.** A PDF parser that returns messy whitespace, then fixed-size splitting on characters, produces chunks that start mid-sentence. Strip whitespace first, then chunk.
- **Not testing the "I don't know" path.** Ask a question you know is NOT in the corpus. If it hallucinates, fix your prompt before shipping.
- **Using L2 distance for unit-normalised vectors without checking.** `fastembed` produces normalised vectors; cosine and L2 give the same ranking but some sqlite-vec/pgvector setups default to L2. Pick one, know which.

## Cost estimate

- **Local embeddings + local SQLite:** $0. Runs entirely on CPU.
- **LLM generation with Claude Haiku 4.5:** about $0.001 per question at typical chunk sizes. 100 questions ≈ $0.10.
- **Alternative with Groq + Llama 4 free tier:** $0. Rate-limited but plenty for a learning project.

## Deliverables

- A single-file or small-package repo with the working CLI.
- A small `docs/` folder of sample content to demo against.
- A `README.md` showing the commands, an example query, and a screenshot or gist of the output.
- A short write-up (300–500 words) in `NOTES.md` on: what chunk size you settled on and why, one retrieval failure you saw and how you diagnosed it, and one change you would make next.

## Going further

Natural next steps after this project:

- Add BM25 (via `rank_bm25`) and RRF fusion — a weekend's worth of work that teaches you hybrid search (project 2 goes deeper on this).
- Swap `fastembed` for OpenAI `text-embedding-3-small` and measure the quality difference on your test questions.
- Add Langfuse tracing with the `@observe()` decorator and look at the traces. You will learn more about your own system in 20 minutes of trace-reading than in a day of code review.
- Port the storage from `sqlite-vec` to pgvector running in Supabase. Same queries, same pipeline, production-ready.

## References

- `fastembed` — the local embedder used here. https://github.com/qdrant/fastembed
- `sqlite-vec` — SQLite extension for vector search. https://github.com/asg017/sqlite-vec
- LangChain `RecursiveCharacterTextSplitter` — the pattern reimplemented in this project. https://python.langchain.com/docs/how_to/recursive_text_splitter/
- Anthropic Python SDK. https://github.com/anthropics/anthropic-sdk-python
- Pinecone, *Chunking strategies for LLM applications*. https://www.pinecone.io/learn/chunking-strategies/
- Module 3 lessons 01 (what is RAG), 02 (embeddings), 03 (vector DBs), 04 (chunking), 05 (eval basics).

---

[← Back to RAG module](../README.md)
