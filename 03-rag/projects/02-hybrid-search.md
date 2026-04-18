# Project 2 — Hybrid Search Engine with Reranking

🟡 **Intermediate** · builds on lessons 05–08

> Vector search misses exact-match queries, BM25 misses synonyms. Combine them with Reciprocal Rank Fusion, add a reranker, and you have the retrieval stack that every production RAG system runs in 2026.

## What you'll build

A search engine that indexes a document collection with *both* dense vectors and BM25, fuses them with Reciprocal Rank Fusion, reranks the top results with Cohere, and serves the whole thing through a Gradio web UI. You will also build a 20-question evaluation set and compare vector-only, BM25-only, hybrid, and hybrid+reranker side-by-side.

The point is to *feel* the difference between the four retrieval modes. When you see an eval spreadsheet where "BM25" finds a chunk "dense" missed and vice versa, the abstract "dense and sparse are complementary" claim from lesson 06 becomes concrete and you stop guessing about which one to use.

## What you'll learn

- BM25 keyword search and how it complements dense retrieval.
- Reciprocal Rank Fusion — implementing it, and why it beats weighted linear combinations.
- Cross-encoder reranking via the Cohere API.
- The two-stage retrieval pattern: retrieve many, rerank, narrow.
- Building and running a retrieval-quality eval (hit@5, MRR).
- Gradio for building quick AI web UIs.
- How to read an eval spreadsheet and know which failure mode to fix.

## Prerequisites

- Python 3.11+
- Project 1 completed (or equivalent — you understand chunking + embedding + retrieval end-to-end).
- Lessons 05 (eval basics), 06 (hybrid search), 07 (reranking), 08 (query understanding) of this module.
- A **free** Cohere API key — go to https://dashboard.cohere.com, sign up, grab a trial key. Free tier is 1000 rerank calls/month, more than enough for this project.
- An LLM provider key for the final answer generation step (Claude, Groq, or Gemini all work).

## Tech stack

- Python 3.11+
- `fastembed` (local) or OpenAI `text-embedding-3-small` for dense embeddings
- `rank_bm25` for BM25 — pure Python, no dependencies
- `cohere` SDK for reranking
- `gradio` for the web UI
- `pydantic` for data models
- `sqlite3` (stdlib) for storing chunks and metadata — or upgrade to pgvector if you prefer
- `anthropic` / `groq` / `openai` for generation

## Setup

```bash
mkdir hybrid-search && cd hybrid-search
python -m venv .venv
source .venv/bin/activate

pip install \
  fastembed \
  rank_bm25 \
  cohere \
  gradio \
  pydantic \
  anthropic
```

```bash
export COHERE_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Starter scaffold

Three files: `retrieval.py`, `eval.py`, `app.py`.

### `retrieval.py`

```python
"""Hybrid retrieval with dense + BM25 + reranker."""

import json
import math
import sqlite3
from pathlib import Path
from typing import Literal

import cohere
import numpy as np
from fastembed import TextEmbedding
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


class Chunk(BaseModel):
    id: int
    text: str
    source: str
    score: float = 0.0


class RetrievalResult(BaseModel):
    mode: Literal["dense", "bm25", "hybrid", "reranker"]
    chunks: list[Chunk]


class HybridRetriever:
    def __init__(self, db_path: Path, corpus: list[Chunk]):
        self.corpus = corpus
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.cohere = cohere.Client()

        self._embeddings = np.array(
            [list(e) for e in self.embedder.embed([c.text for c in corpus])],
            dtype=np.float32,
        )
        # Normalise so cosine = dot product
        self._embeddings /= np.linalg.norm(self._embeddings, axis=1, keepdims=True)

        tokenised = [c.text.lower().split() for c in corpus]
        self._bm25 = BM25Okapi(tokenised)

    # ─── Single-mode retrievers ───────────────────────

    def dense(self, query: str, k: int = 50) -> list[Chunk]:
        qvec = next(iter(self.embedder.embed([query])))
        qvec = qvec / np.linalg.norm(qvec)
        scores = self._embeddings @ qvec
        top = np.argsort(-scores)[:k]
        return [self._clone(self.corpus[i], score=float(scores[i])) for i in top]

    def bm25(self, query: str, k: int = 50) -> list[Chunk]:
        scores = self._bm25.get_scores(query.lower().split())
        top = np.argsort(-scores)[:k]
        return [self._clone(self.corpus[i], score=float(scores[i])) for i in top]

    # ─── Fusion ───────────────────────────────────────

    @staticmethod
    def rrf(ranked_lists: list[list[Chunk]], k: int = 60) -> list[Chunk]:
        scores: dict[int, float] = {}
        chunks: dict[int, Chunk] = {}
        for ranked in ranked_lists:
            for rank, c in enumerate(ranked, start=1):
                scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (k + rank)
                chunks[c.id] = c
        ordered = sorted(chunks.values(), key=lambda c: scores[c.id], reverse=True)
        return [c.model_copy(update={"score": scores[c.id]}) for c in ordered]

    def hybrid(self, query: str, k: int = 20) -> list[Chunk]:
        dense = self.dense(query, k=50)
        sparse = self.bm25(query, k=50)
        return self.rrf([dense, sparse])[:k]

    # ─── Two-stage retrieve + rerank ──────────────────

    def rerank(self, query: str, candidates: list[Chunk], top_n: int = 5) -> list[Chunk]:
        docs = [c.text for c in candidates]
        result = self.cohere.rerank(
            model="rerank-v3.5",
            query=query,
            documents=docs,
            top_n=top_n,
        )
        out = []
        for r in result.results:
            chunk = candidates[r.index].model_copy(
                update={"score": float(r.relevance_score)}
            )
            out.append(chunk)
        return out

    def full(self, query: str, top_n: int = 5) -> list[Chunk]:
        candidates = self.hybrid(query, k=20)
        return self.rerank(query, candidates, top_n=top_n)

    # ─── Utility ──────────────────────────────────────

    @staticmethod
    def _clone(c: Chunk, score: float) -> Chunk:
        return c.model_copy(update={"score": score})
```

### `eval.py`

```python
"""Retrieval eval harness — hit@k and MRR for all four modes."""

import json
from pathlib import Path

from retrieval import Chunk, HybridRetriever

EVAL_SET = Path("eval/questions.json")


def hit_at_k(retrieved_ids: list[int], expected_ids: list[int], k: int) -> int:
    return int(any(i in expected_ids for i in retrieved_ids[:k]))


def reciprocal_rank(retrieved_ids: list[int], expected_ids: list[int]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_ids:
            return 1.0 / rank
    return 0.0


def run_eval(retriever: HybridRetriever, eval_set: list[dict], k: int = 5) -> dict:
    modes = {
        "dense":     lambda q: retriever.dense(q, k=k),
        "bm25":      lambda q: retriever.bm25(q, k=k),
        "hybrid":    lambda q: retriever.hybrid(q, k=k),
        "full":      lambda q: retriever.full(q, top_n=k),
    }
    report = {}
    for mode, run in modes.items():
        hits, rrs = 0, []
        for item in eval_set:
            chunks = run(item["query"])
            ids = [c.id for c in chunks]
            hits += hit_at_k(ids, item["expected_chunk_ids"], k)
            rrs.append(reciprocal_rank(ids, item["expected_chunk_ids"]))
        n = len(eval_set)
        report[mode] = {
            "hit_at_k": hits / n,
            "mrr": sum(rrs) / n,
        }
    return report


if __name__ == "__main__":
    import corpus  # you write this: loads your chunks
    retriever = HybridRetriever(db_path=Path("hybrid.sqlite"), corpus=corpus.load())
    eval_set = json.loads(EVAL_SET.read_text())
    report = run_eval(retriever, eval_set, k=5)
    print(json.dumps(report, indent=2))
```

### `app.py`

```python
"""Gradio web UI — toggle between retrieval modes and see what each finds."""

import gradio as gr
from anthropic import Anthropic

from retrieval import HybridRetriever
import corpus

retriever = HybridRetriever(db_path=None, corpus=corpus.load())
client = Anthropic()

SYSTEM = """\
Answer only from the provided context. Cite chunk IDs in brackets like [3].
If the context does not answer the question, say "I don't know based on the provided documents."
"""


def answer(query: str, mode: str, top_k: int):
    if mode == "dense":
        chunks = retriever.dense(query, k=top_k)
    elif mode == "bm25":
        chunks = retriever.bm25(query, k=top_k)
    elif mode == "hybrid":
        chunks = retriever.hybrid(query, k=top_k)
    elif mode == "full":
        chunks = retriever.full(query, top_n=top_k)
    else:
        return "Unknown mode.", ""

    context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }],
    )
    answer = "".join(b.text for b in response.content if b.type == "text")
    source_md = "\n\n".join(
        f"**[{i+1}]** score={c.score:.3f}  source={c.source}\n\n{c.text[:300]}..."
        for i, c in enumerate(chunks)
    )
    return answer, source_md


with gr.Blocks(title="Hybrid Search Demo") as demo:
    gr.Markdown("# Hybrid Search Playground\nCompare retrieval modes on the same query.")
    with gr.Row():
        query = gr.Textbox(label="Question", scale=3)
        mode = gr.Dropdown(
            choices=["dense", "bm25", "hybrid", "full"],
            value="full",
            label="Mode",
        )
        k = gr.Slider(1, 20, value=5, step=1, label="top_k")
    run = gr.Button("Search")
    answer_box = gr.Markdown(label="Answer")
    source_box = gr.Markdown(label="Sources")
    run.click(answer, [query, mode, k], [answer_box, source_box])


if __name__ == "__main__":
    demo.launch()
```

You still need to write `corpus.py` — a module that loads your documents, chunks them, and returns a `list[Chunk]`. Reuse the chunking code from Project 1.

## Must-have requirements

- ✅ Index a corpus of at least 100 chunks with both dense embeddings and BM25.
- ✅ Implement Reciprocal Rank Fusion to combine dense and sparse results (k=60 default).
- ✅ Integrate Cohere Rerank for a two-stage retrieval.
- ✅ Gradio UI with:
  - Text input for queries
  - Dropdown to select mode: `dense`, `bm25`, `hybrid`, `full` (hybrid + rerank)
  - Slider for top-k
  - Display of retrieved chunks with scores
  - LLM-generated answer that cites chunks
- ✅ An eval set of at least 20 `(query, expected_chunk_ids)` pairs.
- ✅ Eval harness that reports hit@5 and MRR for all four modes, as a table.
- ✅ Log which mode won per query — useful for understanding the complementarity of dense and sparse.

## Stretch goals (pick at least one)

- **Query expansion.** Add a query rewriter that generates 2 alternative phrasings and fuses all three into the dense retrieval step.
- **Feedback loop.** Let users click thumbs-up / thumbs-down on each result; log the feedback and use it to build a labelled training set for future improvements.
- **Multiple doc types.** Support PDFs (via `pdfplumber`) and HTML (via `beautifulsoup4`) alongside markdown. Extract tables from PDFs as separate chunks.
- **Local reranker fallback.** When `COHERE_API_KEY` is missing, fall back to a local `BAAI/bge-reranker-v2-m3` via `sentence-transformers`. Handy for offline demos.

## Rubric

| Area | 1 point | 2 points | 3 points |
|---|---|---|---|
| **Indexing** | Builds both indexes | + Parallelised / efficient | + Incremental, schema clean |
| **Retrieval** | 4 modes work | + RRF correctly implemented (verified by eval) | + Reranker measurably improves MRR |
| **UI** | Gradio demo runs | + Source display with scores | + Per-query mode comparison view |
| **Eval** | hit@5 + MRR printed | + Per-mode table | + Failure bucket inspected and analysed |
| **Code quality** | Works | + Pydantic models, typed retrieval interface | + Separation of concerns; tests for RRF and hit@k |

Target: 12+ / 15. If your eval shows that `full` mode is NOT the best on some queries, that is interesting — dig into those failures, they teach you more than the aggregate.

## Common pitfalls

- **BM25 tokenisation.** `rank_bm25` uses whatever tokenisation you hand it. `text.lower().split()` is the minimum; for better quality, remove punctuation or use `nltk.word_tokenize`. Mismatched tokenisation between index and query is a silent killer.
- **Score scales.** Do not try to combine raw dense and BM25 scores linearly. You will spend an afternoon tuning weights and RRF will still beat you. Use RRF.
- **Reranker sees too few candidates.** Retrieving only 5 candidates and then reranking gives the reranker nothing to work with. Retrieve 20–50, then narrow.
- **Forgetting the `k` in RRF.** The 60 is not magic but it is the standard; if you change it to 1, the formula breaks in unexpected ways.
- **Eval set too clean.** If all your eval queries are perfectly formed and keyword-rich, BM25 will look better than hybrid because dense search has nothing to add. Mix in conversational queries too.
- **Eval set too small.** 10 questions is not enough to discriminate; 20 is the minimum, 50+ is much better.

## Cost estimate

- **Local embeddings + `rank_bm25`:** $0.
- **Cohere Rerank free tier:** 1000 calls/month. Each test query is 1 call; 20 eval questions × 4 modes = 80 calls per eval run. You can run 12+ eval runs before hitting the free limit.
- **LLM generation with Haiku:** $0.001–$0.002 per question.
- **Total for a full build-and-evaluate cycle:** under $0.50.

## Deliverables

- Repo with `retrieval.py`, `eval.py`, `app.py`, `corpus.py`, and a `README.md`.
- `eval/questions.json` with 20+ labelled queries.
- A `NOTES.md` with:
  - The final eval table (hit@5 + MRR for all four modes).
  - A section called "3 queries where BM25 beat dense" with the queries and a sentence on why.
  - A section called "3 queries where dense beat BM25" with the same.
  - The single biggest improvement you saw from adding the reranker, with before/after scores.
- A short screen recording or screenshot of the Gradio UI.

## Going further

- Replace `rank_bm25` + in-memory vectors with **Postgres full-text search + pgvector** in a single table. This is the production pattern and works unchanged up to a few million chunks.
- Add **contextual chunking** (lesson 09) to the ingestion step. Measure the hit@5 and MRR uplift — this is the best way to build intuition for how much it actually helps.
- Add **Langfuse tracing** (`@observe()` on the retrieval functions) so you can watch production queries and spot patterns.
- Swap Cohere Rerank for **Voyage rerank-2.5** and **`bge-reranker-v2-m3`** and compare all three in a fourth mode column.

## References

- Cohere Rerank docs. https://docs.cohere.com/docs/rerank-overview
- `rank_bm25` — pure-Python BM25. https://github.com/dorianbrown/rank_bm25
- Cormack, Clarke & Büttcher 2009, *Reciprocal Rank Fusion*. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Qdrant, *Hybrid search*. https://qdrant.tech/articles/hybrid-search/
- Anthropic, *Contextual Retrieval* — the end-of-pipeline target. https://www.anthropic.com/news/contextual-retrieval
- Pinecone, *Rerankers explained*. https://www.pinecone.io/learn/series/rag/rerankers/
- Gradio docs. https://www.gradio.app/docs
- Module 3 lessons 05 (eval), 06 (hybrid search), 07 (reranking), 08 (query understanding).

---

[← Back to RAG module](../README.md)
