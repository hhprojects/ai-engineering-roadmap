# LLM API Gateway with Caching

🟡 **Intermediate**

Every production AI app needs a proxy layer between your code and LLM APIs. Build one with semantic caching, rate limiting, cost tracking, and automatic model fallbacks.

## What You'll Build

A proxy service that sits in front of LLM APIs using LiteLLM, with semantic caching (return cached responses for similar prompts), per-user rate limiting, cost tracking, and automatic retries with model fallback.

## What You'll Learn

- Building API proxy/gateway patterns
- Semantic caching with embeddings
- Per-user rate limiting and cost tracking
- Model fallback and retry strategies
- LiteLLM for unified multi-provider access

## Tech Stack

- Python 3.11+
- FastAPI
- LiteLLM
- ChromaDB or Redis for cache
- `fastembed` for cache embeddings
- SQLite for cost tracking
- Docker

## Requirements

- Build a FastAPI proxy that accepts OpenAI-compatible chat completion requests
- Route requests through LiteLLM to any supported model
- Implement semantic caching:
  - Embed incoming prompts
  - If a similar prompt (cosine similarity > 0.95) exists in cache, return the cached response
  - Cache hit/miss should be visible in response headers
- Per-user rate limiting (configurable limits per API key)
- Cost tracking:
  - Log token usage and estimated cost per request
  - Track cumulative cost per user/API key
  - Endpoint to query current spend
- Automatic retry with model fallback:
  - If primary model fails, try next tier (e.g., GPT-4o → GPT-4o-mini → Groq)
  - Configurable fallback chain
- Include a `/stats` endpoint showing: cache hit rate, total requests, cost breakdown
- Docker containerized with `docker-compose.yml`
- At least 20 tests covering caching, rate limiting, and fallback behavior

## Stretch Goals

- Add a cost alert system (email/webhook when daily spend exceeds threshold)
- Build a real-time cost dashboard with Streamlit
- Implement request queuing for burst traffic

## Hints

- LiteLLM already handles multi-provider routing — don't reinvent it, just add your caching and tracking layers on top
- For semantic caching, the cache key is the prompt embedding, not the raw text. Use a similarity threshold, not exact match.
- `fastapi.middleware` is the right place for rate limiting and cost tracking — keep route handlers clean

---

[← Back to Production](../README.md)
