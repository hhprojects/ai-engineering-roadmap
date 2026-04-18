# 03 — Vector Databases

> A vector database is a specialised index that answers "find me the `k` vectors closest to this one" in milliseconds instead of seconds, plus the bookkeeping (metadata, filters, updates, persistence) that a production RAG pipeline actually needs.

Every RAG tutorial starts with a vector database because you literally cannot do retrieval without one, but most tutorials gloss over the only interesting question: **which one, and why?** The answer in 2026 is different from the answer in 2023. Postgres caught up, managed services got cheaper, and the "dedicated vector DB vs. Postgres" debate has softened into "it depends on your ops constraints." This lesson gives you the decision framework plus enough HNSW intuition to actually tune your index.

## What a vector DB does that a loop cannot

For a few thousand chunks, `numpy` works fine. You load the vectors into a `(n, d)` array, matrix-multiply with the query vector, and take the top-k. This is exact search — you are guaranteed to find the closest neighbours — and it is around 1 ms per query for small corpora. Go install `scikit-learn` and try it; you will be amazed how far you can get.

The problem is that exact search is `O(n × d)` per query. At 10 million chunks of 1536 dimensions, each query is comparing 15 billion floats. Even with SIMD tricks, that is hundreds of milliseconds per query and an enormous CPU bill. Vector databases solve this with **approximate nearest neighbour (ANN)** indexes, which return the top-k neighbours with ~95–99% recall (you might miss the actual closest vector, but you find the next-closest) in sublinear time.

A good vector DB gives you five things:

1. **An ANN index** (HNSW, IVF, or a variant) with sane defaults.
2. **Metadata filters** — "only search chunks where `tenant_id = 42` and `language = 'en'`."
3. **Incremental updates** — add, update, and delete vectors without rebuilding the index.
4. **Persistence** — survives restarts, handles crashes, supports backups.
5. **Operational features** — replication, horizontal scaling, authentication, rate limiting.

You can write all of this yourself on top of FAISS, and many teams have. You will regret it the first time a customer asks for deletion under GDPR while your index is being rebuilt.

## HNSW: the index you will actually use

Hierarchical Navigable Small World (HNSW) is the default index in almost every modern vector DB — pgvector, Qdrant, Weaviate, Chroma, Milvus, Pinecone. It is worth understanding in rough terms because its three tuning parameters show up everywhere.

**The intuition:** HNSW builds a multi-layer graph. The bottom layer contains every vector, connected to its nearest neighbours. Higher layers contain exponentially fewer vectors, with longer jumps between them. Search starts at the top (a small number of vectors, big jumps), follows the greedy path toward the query, then descends into denser layers until it reaches the bottom and returns the `k` closest nodes it found. Think of it as skip lists for high-dimensional space.

**The three parameters:**

- **`m`** (default 16): maximum connections per node per layer. Larger `m` means denser graph, better recall, more memory. 16 is the right default for most corpora; bump to 32 or 48 for high-quality or high-dimension workloads.
- **`ef_construction`** (default 64–200): how many candidates to consider when *building* the graph. Higher is slower to index but gives a better graph. Most DBs default to 64; 200 is worth it if index time is not a constraint.
- **`ef_search`** (default 40–100): how many candidates to consider at query time. This is the knob you tune at runtime — higher `ef_search` means higher recall and higher latency. Start at 40, then raise until recall plateaus.

For pgvector specifically: create the index with `CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);` and control recall per query with `SET hnsw.ef_search = 100;`. Default recall is around 95% for most corpora; most production systems target 98% and accept the extra few milliseconds per query.

**Memory cost:** HNSW keeps the graph in RAM. Roughly, you pay `n × d × 4 bytes` for the vectors plus `n × m × 16 bytes` for the graph edges. A 1 million chunk × 1536 dim corpus with `m=16` is about 6 GB of RAM. Budget accordingly.

## When HNSW is not the right index

Two situations:

- **Billion-scale corpora** where RAM is the bottleneck. IVF-PQ (inverted file + product quantisation) compresses each vector to a few bytes at a significant recall cost. Milvus, FAISS, and Qdrant support it.
- **Small, static corpora** (< 100k vectors) where exact search is simpler. `IVFFlat` or brute force gets you 100% recall with less tuning. pgvector's `IVFFlat` index is perfectly fine at this scale.

Do not reach for quantisation or IVF-PQ until you have measured and confirmed that HNSW cannot fit. Modern RAM is cheap; developer time tuning quantisation is not.

## The vector DB landscape in 2026

There are maybe thirty vector databases. Here are the six you will actually consider:

| Database | Deployment | Best for | Watch out for |
|---|---|---|---|
| **pgvector** | Self-host or Supabase / Neon / RDS | Teams already running Postgres. Small-to-medium scale (< 10M vectors). Transactional consistency with the rest of your data. | Memory budget for HNSW. Query planner sometimes picks bad plans — use `SET enable_seqscan = off` when debugging. |
| **Qdrant** | Self-host (Rust) or cloud | Open source with production-grade features. Strong hybrid search and filtering. Fast on a single node. | Smaller ecosystem than the incumbents; docs occasionally lag releases. |
| **Pinecone** | Managed only | Zero ops. Serverless scales to billions. Namespaces for multi-tenancy. Pod-based and serverless tiers. | Lock-in. Cost at scale. No self-host option. |
| **Weaviate** | Self-host or managed | Batteries-included — hybrid search, generative modules, multi-tenant isolation built in. | Heavier resource footprint than Qdrant; steeper learning curve. |
| **Chroma** | Embedded or hosted | Prototypes, notebooks, "just works" developer experience. SQLite-backed local mode. | Not the right pick once you need multi-node or heavy filtering. Weak at scale. |
| **Milvus** | Self-host or Zilliz Cloud | Massive scale (hundreds of millions to billions of vectors). GPU acceleration. | Operational complexity. Overkill for anything under ~50M chunks. |

The 2026 decision tree I actually use:

- **Prototype on my laptop:** Chroma or `fastembed` + `numpy`.
- **Production, already running Postgres:** pgvector, no question. The operational simplicity of one database is worth a lot.
- **Production, no Postgres or very large scale:** Qdrant self-hosted for cost control, Pinecone for zero-ops.
- **Multi-tenant SaaS with thousands of isolated customer indexes:** Pinecone namespaces, Weaviate multi-tenancy, or Qdrant collections — whichever your team can operate.
- **Billion-scale, on-prem, performance-critical:** Milvus with GPU acceleration.

Do not treat this as dogma. The differences between the top five are smaller than the differences caused by your chunking strategy and your reranker choice. Pick one that fits your ops constraints, build your eval harness, and spend your optimisation budget upstream of the index.

## Metadata filters matter more than you think

Almost every RAG system needs to filter results by something — user ID, document type, date, language, tenant. The way a vector DB implements filters dramatically affects correctness and latency. There are two strategies:

- **Pre-filter:** Filter the corpus first, then ANN-search the filtered subset. Perfect recall but can be very slow if the filter is selective, because the ANN graph is built on the full corpus.
- **Post-filter:** ANN-search the full corpus, then drop results that fail the filter. Fast but the filter can empty your top-k completely if the matching chunks happen to be far from the query in embedding space.

Qdrant, Pinecone, and Weaviate implement **filterable HNSW** — they build the graph with payload filters as first-class citizens, so the search walks only nodes matching the filter. This is the right implementation for most workloads and a good reason to prefer these DBs for heavy filtering.

pgvector uses pre-filter (the SQL `WHERE` clause) combined with HNSW, and it works well provided you have an index on the filter column. Watch out for cases where the planner bails out of the HNSW index and falls back to sequential scan; when that happens, the `ef_search` limit is effectively ignored and you will see surprising latency jumps.

## Storing chunks, not just vectors

A common rookie mistake: storing only the vector in the vector DB and keeping the chunk text in a separate Postgres table, joined at query time. Two problems:

1. You have to do a second round-trip per query, doubling latency.
2. You now have two sources of truth for your chunks and a consistency bug waiting to happen.

Every vector DB supports a **payload** or **metadata** field — use it. Store the chunk text, source URL, title, page number, and anything else you will need to display or filter on, alongside the vector. Your retrieval call returns everything the frontend needs in one round-trip.

For pgvector this is trivial because it is Postgres: one table, one query, no join:

```sql
CREATE TABLE chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id UUID NOT NULL,
  chunk_text TEXT NOT NULL,
  chunk_index INT NOT NULL,
  embedding vector(1536) NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX chunks_embedding_idx
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX chunks_document_idx ON chunks(document_id);
CREATE INDEX chunks_metadata_idx ON chunks USING gin (metadata);
```

That schema handles metadata filtering, document deletion, and atomic updates for free.

## Sizing and cost

A rough back-of-envelope for a 1 million chunk corpus at 1536 dims with OpenAI `text-embedding-3-small`:

- **Embedding cost (one-time):** ~500M tokens × $0.02 / 1M = **$10**.
- **Storage (vectors only):** 1M × 1536 × 4 bytes ≈ **6 GB**.
- **Storage (vectors + HNSW graph):** roughly **9–10 GB RAM**.
- **pgvector on Supabase or Neon:** comfortably fits on a medium instance (~$25–50/month).
- **Pinecone serverless:** ~$10–20/month for this size at moderate query volume.
- **Qdrant self-hosted:** one small VM, ~$15/month.

The storage is cheap. The embedding compute for an initial index is trivial. The recurring costs at this scale are query volume (if managed) and compute (if self-hosted), and both are dwarfed by the LLM generation cost per query. Do not over-optimise the vector DB bill; optimise the LLM prompt size instead.

## What to remember

- ANN is why vector DBs exist; HNSW is the index you will almost always use.
- Tune HNSW with `m`, `ef_construction`, and `ef_search`. `ef_search` is the runtime knob.
- If you already run Postgres, start with pgvector. The operational simplicity wins unless you have a specific reason to choose otherwise.
- Store chunk text and metadata *with* the vector, not in a separate table you have to join.
- Metadata filters matter — pick a DB with filterable HNSW if you filter heavily.
- Vector DB cost is trivial compared to LLM generation cost. Do not micro-optimise the index.

## References

- `pgvector` — the open-source Postgres extension. https://github.com/pgvector/pgvector
- Malkov & Yashunin 2016, *HNSW — Efficient and robust approximate nearest neighbor search*. https://arxiv.org/abs/1603.09320
- Qdrant, *Hybrid search with dense and sparse vectors*. https://qdrant.tech/articles/hybrid-search/
- Weaviate, *Hybrid search explained*. https://weaviate.io/blog/hybrid-search-explained
- Pinecone, *What is a vector database?* https://www.pinecone.io/learn/vector-database/
- Supabase, *pgvector performance and indexes*. https://supabase.com/docs/guides/ai
