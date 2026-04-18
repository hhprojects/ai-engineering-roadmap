# 02 — Embeddings and Semantic Search

> An embedding is a learned coordinate system where semantically similar text ends up close together, so "nearest neighbour" becomes a usable proxy for "most relevant."

Embeddings are the math trick that makes RAG possible. If you understand exactly what they capture, what they do not capture, and how to pick one, the rest of retrieval becomes mostly engineering. If you treat them as a black box, you will spend a lot of time debugging "why didn't my search find the obvious chunk?" — the answer is almost always that you asked the embedding model a question it was never designed to answer.

## What an embedding actually is

An embedding is a fixed-length vector of floating-point numbers — typically 384, 768, 1024, 1536, or 3072 dimensions — that a trained neural network assigns to a piece of text. Two things matter about it:

1. **It is learned.** A transformer-based encoder is trained on hundreds of millions of text pairs with a contrastive objective: pull semantically related texts close together in the vector space, push unrelated ones apart.
2. **It is the output of a fixed model.** Given the same text and the same model version, you get the same vector, forever. That is what makes pre-computation possible.

Once you have a model that assigns vectors to text, you can compute the similarity of any two pieces of text by comparing their vectors with a cheap numerical operation — cosine similarity or dot product — rather than running a neural network on the pair. That asymmetry (one expensive encode per text, many cheap comparisons afterwards) is what lets vector databases scale.

A useful mental model: imagine projecting every sentence in the English language onto a map where "semantic meaning" is the distance metric. "The cat sat on the mat" and "A feline rested on the rug" land next to each other; "Consider a variable named `sum`" lands in a completely different neighbourhood. The embedding model is that projection function.

## Cosine similarity, dot product, L2

Three distance functions show up constantly in RAG:

- **Cosine similarity:** the cosine of the angle between two vectors. Returns a value in `[-1, 1]`, where 1 means identical direction. This ignores vector magnitude, which is what you want if the embedding model was trained with a normalised objective.
- **Dot product:** vectors multiplied component-wise and summed. For unit-length vectors this is identical to cosine similarity. For un-normalised vectors, dot product also rewards magnitude, which can be useful when the model was trained to encode "confidence" in vector length.
- **L2 (Euclidean) distance:** straight-line distance in the vector space. For unit-length vectors, this is monotonically related to cosine similarity — the rankings are identical — so most vector DBs let you pick whichever is faster for your index.

The practical advice: **read the model card**. OpenAI `text-embedding-3-small`, Cohere `embed-v4`, Voyage `voyage-3`, and most open-source models (BGE, E5, Nomic) produce normalised vectors and expect cosine or dot product. A few older models are not normalised and expect L2. Using the wrong metric quietly degrades quality; it will not error, it will just return worse rankings.

## The 2026 embedding model landscape

As of early 2026, these are the models you actually pick between:

| Model | Dims | Strength | Cost | Notes |
|---|---|---|---|---|
| OpenAI `text-embedding-3-small` | 1536 (Matryoshka) | Cheap, fast, good general English | $0.02 / 1M tokens | Can truncate to 512 or 256 dims for storage savings |
| OpenAI `text-embedding-3-large` | 3072 (Matryoshka) | Best OpenAI quality | $0.13 / 1M tokens | Overkill for most use cases |
| Cohere `embed-v4` | 1536 | Multilingual, strong reranker pairing | $0.10 / 1M tokens | Pairs naturally with Cohere Rerank |
| Voyage `voyage-3` / `voyage-3-lite` | 1024 / 512 | Top of the MTEB leaderboard, very good retrieval | $0.06–$0.12 / 1M tokens | Anthropic's recommended pairing for contextual retrieval |
| Google `text-embedding-004` | 768 | Integrated with Vertex, strong multilingual | Tiered | Good default inside GCP |
| `bge-large-en-v1.5` | 1024 | Best open-source English, runs local | Free + GPU | `fastembed` ships this; good for prototypes and privacy |
| `nomic-embed-text-v1.5` | 768 (Matryoshka) | Open weights, long context (8k tokens) | Free + GPU | The open-source "does it all" pick |
| `gte-multilingual-base` / `multilingual-e5-large` | 768 / 1024 | Multilingual open-source | Free + GPU | Use when you need 50+ languages on local hardware |

**Matryoshka Representation Learning (MRL)** is worth understanding because it keeps showing up: a model is trained so that the first `k` dimensions of its output vector are themselves a valid (slightly lower-quality) embedding. OpenAI's `-3` series, Nomic, and many 2025+ models ship with MRL, which lets you store 256 or 512 dims per chunk instead of 1536 and cut your storage bill by 3–6× with only a small quality hit. Always check whether your model supports this before provisioning a vector DB.

## What embeddings can and cannot do

The failure modes of semantic search all come from the same place: the embedding model has learned a projection that is *good* at "overall topical similarity" and *bad* at anything that depends on small textual differences. Concretely:

**Embeddings handle well:**
- Paraphrase: "How do I reset my password?" vs. "I forgot my login credentials, help."
- Synonym overlap: "automobile insurance" vs. "car policy."
- Cross-topic bridging: "Which document covers GDPR compliance?" vs. a chunk titled "Data protection regulations."
- Multilingual alignment (with multilingual models).

**Embeddings handle badly:**
- **Exact identifiers.** Product SKUs, error codes, ticker symbols, invoice numbers. `ERR-4472` and `ERR-4473` are indistinguishable to a dense encoder. Fix with BM25 / hybrid search (lesson 06).
- **Negation.** "The service is not down" and "The service is down" often embed to nearly-identical vectors. Embedding models mostly ignore negation. Fix with reranking (lesson 07) or explicit negation handling in queries.
- **Small edits that change meaning.** "I paid $100" vs. "I was paid $100" — a bi-encoder smears both into the same neighbourhood.
- **Long documents compressed into one vector.** Losing half the meaning is the point of the compression; it just means you need to chunk (lesson 04) so each vector represents something coherent.
- **Out-of-domain jargon.** A general-purpose model that has never seen your internal acronyms will put them in random positions. Fix with fine-tuning the embedder on your domain, or with hybrid search.

The rule of thumb: **if a human would find the answer by scanning for a specific token, use BM25 or hybrid. If a human would find it by understanding what the question is about, use embeddings.** Most real workloads need both, which is why hybrid retrieval is a lesson on its own.

## How to pick an embedding model

A practical decision process, in order:

1. **Start with a free model to prototype.** `fastembed` ships `BAAI/bge-small-en-v1.5` out of the box; it runs on CPU, has no API key, and is perfectly adequate for the first few hundred documents. Use it until your corpus or your quality bar demand more.
2. **Read the model card for your target language(s).** If you need Mandarin, Bahasa, or any low-resource language, check MTEB and the model card. Multilingual models (Cohere `embed-v4`, `multilingual-e5-large`) are much better than English models on non-English input.
3. **Match your eval set against two or three candidates.** Do not trust MTEB scores blindly. Build a 50-question eval with known-correct documents (this is lesson 05), then measure hit@5 for each candidate on your actual data. A model that is third on MTEB can be first on your specific domain.
4. **Consider pairing.** Anthropic's contextual retrieval paper used Gemini and Voyage embeddings together with Voyage or Cohere rerank. If you plan to rerank, pick an embedding model that has a good matching reranker — the pairing often matters more than the absolute embedder quality.
5. **Factor in migration cost.** Embeddings are sticky: changing the model means re-embedding the entire corpus and invalidating the vector index. Budget for one migration during the project's lifetime; choose a model you are happy to live with for 6–12 months.

## The encoding API

All the major providers look the same at the API level. OpenAI's endpoint:

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["chunk one", "chunk two", "chunk three"],
)
vectors = [e.embedding for e in response.data]
```

A few practical notes:

- **Batch aggressively.** All providers accept up to 2048 inputs per request. One request for 1000 chunks is vastly cheaper and faster than 1000 requests for 1 chunk. Your index-time pipeline should batch with a backpressure-aware queue.
- **Cache embeddings by text hash.** If you re-index the same document twice, pay the embedding cost once. Most production pipelines store a content hash alongside the vector and skip re-embedding when the hash matches.
- **Handle the token limit.** Every model has one — `text-embedding-3-small` is 8192 tokens. Your chunker must enforce it; if it does not, the API will truncate silently and you will embed half a chunk without knowing it.
- **Normalise once.** If your vector DB expects unit vectors and your model does not produce them, normalise at index time, not every query.

## The similarity search step

Once you have vectors for every chunk and a vector for the query, retrieval is a k-nearest-neighbour problem: find the `k` chunks whose vectors are closest to the query vector. Naively, this is `O(n)` per query — compute `n` distances and sort. That is fine up to around 10,000 chunks and a non-starter at a million.

Vector databases (lesson 03) use approximate nearest neighbour (ANN) algorithms — HNSW, IVF, IVF-PQ — to trade a small amount of recall for several orders of magnitude of speed. At query time you typically retrieve 50 to 200 candidates with ANN and then rerank the top 20 with a cross-encoder. This two-stage pattern (lesson 07) is the workhorse of production RAG.

## What to remember

- An embedding is a fixed-length vector that encodes semantic meaning; cosine or dot product measures similarity.
- Read the model card — pick the distance metric the model was trained with.
- Embeddings are good at paraphrase and topic, bad at exact identifiers, negation, and tiny edits. That is why hybrid search exists.
- Start with a free local model (`fastembed` / BGE). Upgrade to Voyage, Cohere, or OpenAI `-3` once you have an eval set that can prove it is worth the cost.
- Matryoshka embeddings let you truncate dimensions to save storage at small quality cost. Use them.
- Batch at index time. Cache by content hash. Never re-embed unchanged chunks.
- Changing embedding model means re-indexing the whole corpus. Budget for one migration per project lifetime.

## References

- OpenAI, *Embeddings guide*. https://platform.openai.com/docs/guides/embeddings
- Cohere, *Embed API documentation*. https://docs.cohere.com/docs/embeddings
- Voyage AI, *voyage-3 announcement* — strong retrieval quality and the default pairing for Anthropic contextual retrieval. https://blog.voyageai.com/
- MTEB leaderboard — public ranking of embedding models on retrieval benchmarks. https://huggingface.co/spaces/mteb/leaderboard
- Kusupati et al. 2022, *Matryoshka Representation Learning*. https://arxiv.org/abs/2205.13147
- `fastembed` — lightweight embeddings without API keys. https://github.com/qdrant/fastembed
- Eugene Yan, *Patterns for building LLM-based systems & products* — the RAG / embeddings section. https://eugeneyan.com/writing/llm-patterns/
