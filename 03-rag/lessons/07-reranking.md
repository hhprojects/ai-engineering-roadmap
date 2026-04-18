# 07 — Reranking

> A reranker is a cross-encoder that reads the query and each candidate chunk *together* and scores their relevance directly; adding one on top of hybrid search is usually the cheapest, highest-leverage quality win in the entire RAG stack.

You have hybrid retrieval producing a top-20 candidate list that is probably decent. Reranking is the step that turns "probably decent" into "reliably correct." In Anthropic's contextual retrieval experiments, adding a reranker on top of contextual embeddings + contextual BM25 dropped the retrieval failure rate from 2.9% to 1.9% — roughly a third fewer failures from a single API call per query. If you are not reranking, you are leaving quality on the table.

## Bi-encoder vs. cross-encoder: the core idea

Every retrieval system so far in this module is a **bi-encoder**: documents and queries are encoded independently, and you compare their vectors with a cheap distance metric. This is what makes vector search scalable — you encode a million chunks once, at index time, and then every query is one more encoding plus a fast ANN lookup.

The cost of that scalability is **information loss**. The encoder has to collapse a 500-token chunk into 1536 floats with no knowledge of what you will later ask. It chooses a generic "what is this chunk about" projection, which is good at finding chunks on the right topic but bad at the fine-grained "given this specific query, which of these two chunks is more relevant?" judgement.

A **cross-encoder** does the opposite trade-off. Instead of encoding documents and queries separately, it takes a `(query, document)` pair as a single input and runs a full transformer over the concatenation. The output is one number: the relevance of that document to that query. Every token of the query can attend to every token of the document, which is why cross-encoders are dramatically more accurate at fine-grained ranking.

The downside is obvious: you cannot pre-compute anything. For every query, you run one transformer inference per candidate. That is why you cannot replace bi-encoders with cross-encoders at retrieval time — running a cross-encoder on 10 million chunks per query would cost dollars and minutes. But running it on 20 candidates per query costs fractions of a cent and milliseconds.

This asymmetry is why every production RAG in 2026 is two-stage:

1. **Retrieve (bi-encoder, fast):** fetch 20–100 candidates with vector search and BM25.
2. **Rerank (cross-encoder, precise):** score each candidate against the query and sort.
3. **Top 3–5 to the LLM.**

## The rerankers you will actually use

### API rerankers (what most teams pick)

- **Cohere Rerank 4 / Rerank 3.5** — the industry standard. 100+ languages, ~50 ms per query for 20 candidates, billed per "search unit" (roughly one query + one batch of documents). Free tier is generous enough for prototyping and small production. Rerank 3.5 is multilingual by default. Cohere publishes their API as `rerank-v3.5` and `rerank-v4`; the v4 series is faster and slightly more accurate.
- **Voyage rerank-2.5 / rerank-2.5-lite** — Anthropic's recommended pairing for contextual retrieval. Strong English quality; competitive with Cohere on most benchmarks.
- **Jina reranker v2** — open-weights, runs locally, smaller quality gap vs. Cohere than you would expect.

API rerankers are the default choice unless you have specific reasons not to use one. They are fast, cheap, and you do not have to provision GPUs.

### Local rerankers (when you cannot send data to an API)

- **`BAAI/bge-reranker-v2-m3`** — open-source multilingual cross-encoder from BAAI. Runs on CPU for small batches, GPU for anything serious. Quality is 90%+ of Cohere.
- **`ms-marco-MiniLM-L-12-v2`** (sentence-transformers) — older but still works, tiny, runs on CPU.
- **`nvidia/NV-RerankQA-Mistral-4B-v3`** — high-quality LLM-as-reranker from Nvidia; heavy but excellent when latency is not critical.

For privacy-sensitive deployments (healthcare, legal, defence) local rerankers are the right answer. For everything else, start with Cohere.

### LLM-as-reranker

You can use a general-purpose LLM as a reranker by asking it to order the candidates. Prompts look like "Here are 20 chunks; list their IDs in order of relevance to this query." It works surprisingly well, especially with strong models (Claude, GPT-5), but:

- It is slow (seconds, not milliseconds).
- It is expensive (tens of thousands of tokens per call).
- It inherits LLM failure modes — prompt injection, output format drift, ordering bias.

Use it for very low-volume or very high-quality-sensitive workloads (research, legal discovery), not for user-facing interactive RAG.

## A concrete reranking example

Cohere is the simplest API to show, so:

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
) -> list[dict]:
    # candidates: [{"id": "c1", "text": "..."}, ...]
    docs = [c["text"] for c in candidates]
    result = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs,
        top_n=top_n,
    )
    reranked = []
    for r in result.results:
        c = candidates[r.index]
        reranked.append({**c, "rerank_score": r.relevance_score})
    return reranked
```

Used in the full pipeline from lesson 06:

```python
candidates = await hybrid(conn, query_text, query_embedding, k=20)
final = rerank_with_cohere(query_text, candidates, top_n=5)
```

Latency: roughly 40–80 ms for 20 candidates on Cohere's v4. Cost: under $0.001 per query at most volumes. The quality delta is usually 5–15 percentage points of MRR on a realistic eval set. It is one of the best ROI layers in the stack.

## Why reranking works so well

Three concrete things that bi-encoder retrieval systematically gets wrong and that cross-encoder reranking systematically fixes:

1. **Topic vs. answer.** A chunk can be about "password reset" without being the chunk that tells you *how* to reset a password. The bi-encoder puts them both at similar distance from the query; the cross-encoder, which reads the query while reading the chunk, distinguishes "this chunk is on-topic but describes a different procedure" from "this chunk directly answers the question."
2. **Negation and polarity.** A bi-encoder often cannot tell "the feature is available" from "the feature is not available"; a cross-encoder usually can, because it sees the query's "do you support X" alongside the chunk's "we do not support X."
3. **Long documents.** Bi-encoder vectors for long chunks are a blurry average of everything in the chunk. The cross-encoder can zoom in on the relevant sentence within the chunk.

You pay for this by running inference per candidate per query, which is why you keep the candidate set small. Typical numbers: retrieve 50–100 with hybrid, rerank to 20, send top 5 to the LLM.

## The "retrieve more, keep less" principle

A subtle but important point: reranking pays for itself more the more candidates you retrieve. Rerank output quality improves as you feed it more candidates (to a point), because the reranker has more options to pick the right one. But feeding it too many raises cost and latency without helping quality, because the bi-encoder's ranking is noise beyond a certain depth.

The sweet spot empirically is:

- **Retrieve:** 30–100 candidates from hybrid. More on large corpora, less on tiny ones.
- **Rerank:** to 10–20.
- **Send to LLM:** 3–10.

Anthropic's contextual retrieval paper specifically called out "top-20 chunks significantly outperformed top-5 or top-10 configurations" — the key was pairing deeper retrieval with reranking, then narrowing again. The failure mode is retrieving too *few* candidates and not giving the reranker enough material to work with.

## Latency budget

A realistic end-to-end budget for a two-stage RAG pipeline:

| Stage | Typical time |
|---|---|
| Query embedding | 20–40 ms |
| Dense ANN (top 50) | 5–30 ms |
| BM25 (top 50) | 5–20 ms |
| RRF fusion | < 1 ms |
| Rerank (top 20) | 40–100 ms |
| LLM generation | 500–3000 ms |
| **Total retrieval (no LLM)** | **~100–200 ms** |

The reranker is the second-slowest step after LLM generation. On high-volume user-facing applications you can:

- Cache reranker results per `(query_hash, candidate_set_hash)` tuple for common queries.
- Use the smaller reranker (`rerank-v3.5` lite, `bge-reranker-v2-m3-base`) when the quality tradeoff is acceptable.
- Parallelise retrieval with `asyncio.gather` so dense and BM25 run concurrently.

Do not skip reranking to save 60 ms. Your LLM generation is 10x longer and the quality improvement is worth it.

## Evaluating reranker changes

Use the same eval harness from lesson 05. Compare MRR and hit@5 across:

- No rerank (hybrid only)
- Small reranker (local MiniLM)
- Large reranker (Cohere rerank-v3.5)
- Top-flight (Cohere rerank-v4 / Voyage rerank-2.5)

A typical progression on a real corpus:

| Config | hit@5 | MRR |
|---|---|---|
| Dense only | 0.71 | 0.48 |
| Hybrid (dense + BM25 + RRF) | 0.81 | 0.56 |
| Hybrid + Cohere rerank-v3.5 | 0.89 | 0.72 |
| Hybrid + Cohere rerank-v4 | 0.91 | 0.76 |

Diminishing returns are real but the gap from "hybrid" to "hybrid + reranker" is almost always larger than the gap between reranker models. Worry about getting *a* reranker in the loop before worrying about which one.

## Common mistakes

- **Reranking without hybrid.** Reranker on top of dense-only retrieval is still better than dense-only, but you are leaving the hybrid win on the table. Add both.
- **Reranking a top-5 list.** There is nothing for the reranker to do. Retrieve 20+ candidates, rerank, then narrow.
- **Using reranker scores as if they were probabilities.** They are not calibrated. Use them only for ordering; do not threshold on them directly without tuning per model.
- **Forgetting to send the actual chunk text.** Reranker APIs need the text, not just the IDs. Double check your pipeline is passing the right thing.
- **Mixing reranker models across languages.** Cohere multilingual handles most languages, but do not assume English-only rerankers will score Chinese chunks correctly.

## What to remember

- Reranking is the highest-ROI single layer in a production RAG stack. Add it early.
- Bi-encoders are for scalable retrieval; cross-encoders are for precise final ordering. Two stages, always.
- Cohere Rerank 3.5 / 4, Voyage rerank, and `bge-reranker-v2-m3` are the three you will pick between.
- Retrieve 30–100 candidates, rerank to ~20, send 3–5 to the LLM. Retrieving too few candidates is a common mistake.
- Expect +0.10 to +0.25 MRR and +5 to +15 hit@5 over hybrid alone on realistic corpora.
- Reranker latency is under 100 ms. It is not a performance problem.

## References

- Cohere, *Rerank overview and API*. https://docs.cohere.com/docs/rerank-overview
- Pinecone, *Rerankers explained*. https://www.pinecone.io/learn/series/rag/rerankers/
- Anthropic, *Introducing Contextual Retrieval* — reranker results section. https://www.anthropic.com/news/contextual-retrieval
- BAAI, `bge-reranker-v2-m3` on Hugging Face. https://huggingface.co/BAAI/bge-reranker-v2-m3
- Voyage AI, *rerank-2.5 announcement*. https://blog.voyageai.com/
- Nogueira & Cho 2019, *Passage re-ranking with BERT* — the original cross-encoder reranker paper. https://arxiv.org/abs/1901.04085
- Qdrant, *Reranking guide*. https://qdrant.tech/articles/reranking/
