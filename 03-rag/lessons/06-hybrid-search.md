# 06 — Hybrid Search: Dense + Sparse (BM25)

> Hybrid search runs a keyword index (BM25) and a vector index in parallel, fuses their results with Reciprocal Rank Fusion, and is the single biggest retrieval quality win you get for almost no additional complexity.

If you add one layer on top of a naive vector-search RAG pipeline, make it hybrid search. Every serious production system does it. The reason is that dense and sparse retrievers fail in *complementary* ways — the cases dense search misses are exactly the cases BM25 nails, and vice versa — so combining them recovers most of the losses from either one alone.

## Why dense alone is not enough

Go back to lesson 02's list of what embeddings are bad at:

- Exact identifiers (`ERR-4472`, `AWS_ACCESS_KEY_ID`, ticket `PROJ-1138`)
- Proper nouns the model has never seen (your startup's internal product name)
- Rare technical terms (domain jargon the embedder was not trained on)
- Exact quotations ("find me the sentence that says 'we are not responsible for'")
- Negation and polarity
- Short queries where semantic meaning is ambiguous

On all of these, a 30-year-old keyword search algorithm beats a 2026 transformer encoder. That is not a flaw of embeddings; it is a property of the problem. Bi-encoders compress 200+ tokens into a single vector, and some information — particularly identifiers and low-frequency tokens — gets blurred out by the compression. A sparse index preserves the identity of each token. The two approaches are complementary, and the best retrieval systems have been hybrid since the 2010s.

## What BM25 is, briefly

BM25 (Best Matching 25) is a family of scoring functions from classical information retrieval that generalises TF-IDF. For a query `Q` and document `D`, the score is roughly:

```
score(D, Q) = Σ over terms q in Q:
    IDF(q) × (f(q, D) × (k1 + 1)) / (f(q, D) + k1 × (1 - b + b × |D| / avgdl))
```

You do not need to memorise the formula, but you should understand the three ideas:

1. **Term frequency:** a document containing "postgres" many times is more relevant to a query about postgres than one that mentions it once — with diminishing returns.
2. **Inverse document frequency:** a term that appears in almost every document (like "the") carries no information. A term that appears in very few documents is a strong signal.
3. **Length normalisation:** long documents have more chances to match any given term, so we normalise by length so a 100-page manual does not trivially beat a one-page summary.

BM25 is built on an inverted index — for each token, the index stores the list of documents containing it plus metadata for scoring. Lookup is `O(log n)` per query term, it is embarrassingly parallel, and it handles typos and partial matches at the tokenisation layer (stemming, n-grams, subword splitting).

The practical libraries you will touch:

- `rank_bm25` — pure Python, 100 lines, perfect for in-memory prototypes and test harnesses.
- OpenSearch / Elasticsearch — battle-tested BM25 at scale; the default when your corpus grows past a few million documents.
- **Postgres full-text search** (`ts_rank_cd` with `tsvector` / `tsquery`) — surprisingly good, lives next to pgvector, deserves to be your first serious BM25 if you already run Postgres.
- Qdrant, Weaviate, Vespa, and Milvus all ship native sparse-vector support so you can run BM25-equivalent scoring inside the same DB as your dense index.

## Reciprocal Rank Fusion: how to combine them

Once you have dense scores and BM25 scores, the natural question is: how do you combine them? You cannot just add them — the two scoring systems have completely different scales (cosine similarity is between 0 and 1; BM25 is unbounded, typically 5 to 40) and the distributions are non-linear in different ways. Weighted linear combinations require careful normalisation per corpus and per query.

**Reciprocal Rank Fusion (RRF)** sidesteps the scaling problem by ignoring scores entirely and using ranks. For each retrieved document, the fused score is:

```
RRF_score(d) = Σ over retrievers r:  1 / (k + rank_r(d))
```

Where `rank_r(d)` is the rank (1-indexed) of document `d` in retriever `r`'s results, and `k` is a small constant (typically 60, from the original Cormack et al. 2009 paper). Documents not retrieved by a given retriever contribute 0.

This has several lovely properties:

- **Scale-free.** Ranks are dimensionless, so you can fuse any number of retrievers with any scoring systems.
- **Robust to outliers.** The reciprocal dampens the effect of very high scores.
- **Simple.** Implemented in 10 lines of Python:

```python
def rrf_fuse(*ranked_lists: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

fused = rrf_fuse(dense_top50, bm25_top50, k=60)[:10]
```

- **Empirically strong.** In the Anthropic contextual retrieval experiments, contextual embeddings + contextual BM25 fused with RRF (no reranker) cut failure rates by 49% over dense alone; Qdrant's docs call RRF "the de facto standard" for hybrid fusion.

`k=60` is a surprisingly stable default. Tuning `k` rarely changes results by more than a percentage point; do not over-optimise it.

## Alternatives to RRF

A few alternatives are worth knowing about but should not be your default:

- **Weighted linear fusion** — normalise both scores to `[0, 1]` then `α × dense + (1-α) × sparse`. Weaviate uses this with `alpha=0.75` as default. It works but requires per-corpus tuning of `α` and is sensitive to score distributions. Use it only if you have an eval harness to tune on.
- **SPLADE** — a neural sparse retriever. It learns which terms to assign to each document (and with what weights) using a transformer, producing a sparse vector where each dimension corresponds to a vocabulary token. You get BM25-style exact-match behaviour with learned term weights and query expansion baked in. Qdrant and Vespa support SPLADE natively; it is excellent but heavier than BM25 at index and query time.
- **ColBERT / late interaction** — stores one vector per token and computes maximum-similarity interactions at query time. Very high quality, very high storage and compute cost. Consider for small, high-precision collections; probably overkill for your first RAG.
- **Cross-encoder reranking only** — skip fusion entirely, retrieve top 100 from dense, rerank. This works but BM25 + dense + rerank is usually better because the two retrievers cover different failure modes before the reranker even sees the candidates.

## The standard hybrid pipeline

Here is the workhorse pattern that shows up in almost every production RAG system in 2026:

```
User query
   │
   ├─► Dense retrieval (top 50)
   │       (embed query, ANN search)
   │
   ├─► BM25 retrieval (top 50)
   │       (tokenise, inverted index lookup)
   │
   ▼
Reciprocal Rank Fusion (dedupe + re-rank by fused score)
   │
   ▼
Top ~20 candidates
   │
   ▼
Cross-encoder reranker (lesson 07)
   │
   ▼
Top 3–5 to the LLM
```

Each retriever pulls 50 candidates (generous, because you are about to filter heavily). RRF dedupes and merges into a list of ~80 unique candidates with fused scores. You take the top 20 and hand them to a reranker for final ordering, then send the top 3 to 5 to the LLM. This pipeline is fast (under 200 ms end-to-end for most sizes), cheap, and significantly better than any single-retriever setup.

## Implementing it on top of pgvector

You can do the whole thing inside Postgres, which is the single biggest operational advantage pgvector has over dedicated vector DBs. The schema from lesson 03 plus a `tsvector` column:

```sql
ALTER TABLE chunks ADD COLUMN chunk_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;

CREATE INDEX chunks_tsv_idx ON chunks USING gin (chunk_tsv);
```

Then two queries, one RRF fusion in application code:

```python
async def dense_top_k(conn, query_embedding, k=50):
    rows = await conn.fetch(
        """
        SELECT id FROM chunks
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        query_embedding, k,
    )
    return [r["id"] for r in rows]

async def bm25_top_k(conn, query_text, k=50):
    rows = await conn.fetch(
        """
        SELECT id
        FROM chunks
        WHERE chunk_tsv @@ plainto_tsquery('english', $1)
        ORDER BY ts_rank_cd(chunk_tsv, plainto_tsquery('english', $1)) DESC
        LIMIT $2
        """,
        query_text, k,
    )
    return [r["id"] for r in rows]

async def hybrid(conn, query_text, query_embedding, k=20):
    dense, sparse = await asyncio.gather(
        dense_top_k(conn, query_embedding, 50),
        bm25_top_k(conn, query_text, 50),
    )
    fused = rrf_fuse(dense, sparse, k=60)
    return fused[:k]
```

For a Qdrant or Weaviate native hybrid, the syntax is different but the pattern is identical: two retrievers, RRF fusion, top N for the next stage.

## Tuning hybrid search

A short tuning checklist, once the baseline is running:

1. **Candidate count per retriever.** Start at 50 each. If hit@5 on your eval improves noticeably at 100 each, the retrievers are too noisy at 50; usually diminishing returns after 100.
2. **Tokeniser / stemming.** Default English stemmers are fine for English; for CJK, Arabic, or mixed-script corpora, use a language-specific tokeniser or you will lose recall.
3. **Stop words.** Default stop word lists are fine; do not tune unless you have a specific case.
4. **RRF `k` constant.** Leave at 60. Seriously.
5. **Fusion weights (if using weighted linear):** tune on the eval set, not by intuition.
6. **Dedup before reranking.** If a chunk appears in both dense and sparse results, you want one copy with the fused score, not two.
7. **Contextual BM25.** If you are using contextual chunking (lesson 09), feed the *contextualised* text to BM25 too, not just the raw chunk. Anthropic's results showed this doubled the dense-only improvement.

## When hybrid does not help

A few cases where hybrid search is not the right focus:

- **Pure narrative prose with no identifiers.** A novel, a collection of blog posts on one topic. Dense alone is often 90% of the way; hybrid adds 1–2 points.
- **Queries that are all the same format.** If every query is a full sentence, dense dominates. If every query is a keyword, BM25 dominates. Hybrid shines when query styles vary.
- **Tiny corpora.** Under a few thousand chunks, reranking the entire corpus end-to-end is viable and beats ANN + fusion.

For everything else — mixed-corpus, mixed-query, real-world RAG — hybrid is the default.

## What to remember

- Dense and sparse retrievers fail on *different* queries. Hybrid search recovers most of those losses for very little cost.
- BM25 is a solved problem; `rank_bm25`, Postgres `tsvector`, OpenSearch, or native vector-DB sparse support all work fine.
- Reciprocal Rank Fusion is the default combination strategy. `k=60`, no need to tune.
- The canonical pipeline: top-50 dense + top-50 sparse → RRF → top-20 → reranker → top-3 to LLM.
- If you use contextual chunking, run BM25 on the contextualised text too.
- Measure hybrid vs. dense on your eval set. Expect +5 to +15 percentage points of hit@5 on typical corpora.

## References

- Cormack, Clarke & Büttcher 2009, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods* — the original RRF paper. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Anthropic, *Introducing Contextual Retrieval* — contextual embeddings + contextual BM25 + RRF results. https://www.anthropic.com/news/contextual-retrieval
- Qdrant, *Hybrid search explained*. https://qdrant.tech/articles/hybrid-search/
- Weaviate, *Hybrid search explained*. https://weaviate.io/blog/hybrid-search-explained
- Formal, Piwowarski & Clinchant 2021, *SPLADE — Sparse Lexical and Expansion Model*. https://arxiv.org/abs/2107.05720
- Eugene Yan, *Patterns for building LLM-based systems* — hybrid retrieval section. https://eugeneyan.com/writing/llm-patterns/
- Postgres `ts_rank_cd` / `tsvector` documentation. https://www.postgresql.org/docs/current/textsearch.html
