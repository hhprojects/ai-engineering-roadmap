# Hybrid Search Engine

🟡 **Intermediate**

Vector search misses keyword matches. Keyword search misses semantic meaning. Combine them and add reranking — that's how production RAG systems actually work.

## What You'll Build

A search engine that combines vector similarity search with BM25 keyword search, reranks results using a cross-encoder or Cohere's reranker, and serves it all through a simple Gradio web UI.

## What You'll Learn

- BM25 keyword search and how it complements vector search
- Hybrid search fusion strategies (Reciprocal Rank Fusion)
- Reranking with cross-encoders or API-based rerankers
- Building web UIs with Gradio
- Comparing retrieval quality with metrics

## Tech Stack

- Python 3.11+
- ChromaDB for vector search
- `rank_bm25` for keyword search
- Cohere Rerank API (free tier) or `sentence-transformers` cross-encoder
- Gradio for web UI
- `fastembed` or OpenAI embeddings

## Requirements

- Index a document collection with both vector embeddings and BM25
- Implement Reciprocal Rank Fusion (RRF) to combine vector and BM25 results
- Add a reranking step using Cohere's reranker (free tier: 1000 calls/month) or a local cross-encoder
- Build a Gradio web interface with:
  - Text input for queries
  - Toggle between vector-only, BM25-only, and hybrid modes
  - Display retrieved chunks with relevance scores
  - Show the LLM-generated answer
- Create a test dataset of at least 20 query/expected-document pairs
- Compare retrieval quality (hit rate, MRR) across the three modes
- Log and display which mode performed best per query

## Stretch Goals

- Add query expansion (rephrase the query before searching)
- Implement a feedback loop — let users mark results as relevant/irrelevant
- Support different document types (PDF, HTML) alongside markdown

## Hints

- Reciprocal Rank Fusion is simple: for each result, score = Σ(1 / (k + rank)) across retrieval methods. k=60 is a common default.
- Cohere's free tier is enough for development — sign up at `dashboard.cohere.com`
- Gradio's `gr.Interface` gets you a working UI in ~10 lines of code. Start there, then customize.

## Cost Estimate

Free with fastembed + Groq + Cohere free tier.

---

[← Back to RAG](../README.md)
