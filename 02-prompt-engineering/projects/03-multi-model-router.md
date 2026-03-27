# Multi-Model Router

🟠 **Advanced**

Not every query needs GPT-4. Build a smart router that sends easy questions to cheap models and hard ones to premium models — the same pattern production AI companies use to cut costs.

## What You'll Build

A routing system that classifies incoming queries by complexity and type, routes simple ones to fast/cheap models (Groq/Llama) and complex ones to GPT-4/Claude, and tracks cost savings vs. quality trade-offs with an evaluation framework.

## What You'll Learn

- Query classification and complexity estimation
- Multi-provider API abstraction
- Cost optimization strategies for LLM applications
- Building evaluation frameworks to measure quality vs. cost
- Production patterns used by AI companies

## Tech Stack

- Python 3.11+
- `openai`, `anthropic`, `groq` SDKs
- `litellm` (optional, for unified API)
- Pydantic v2
- pytest
- SQLite for logging

## Requirements

- Implement a query classifier that categorizes requests (simple factual, creative, code, reasoning, etc.)
- Route queries to appropriate models based on classification:
  - Simple/factual → Groq (Llama 3) or GPT-4o-mini
  - Creative/nuanced → Claude 3.5 Sonnet
  - Complex reasoning → GPT-4o or Claude 3 Opus
- Log every request: query, classification, model used, response, latency, cost
- Build an eval comparing routed responses vs. always-using-premium (quality + cost)
- Include a confidence threshold — if the classifier isn't confident, default to premium
- Support manual override ("always use Claude for this user")
- Calculate and display cumulative cost savings
- Config-driven routing rules (YAML or JSON)
- At least 50 test queries covering all categories

## Stretch Goals

- Implement automatic model fallback (if primary model errors, try next tier)
- Add semantic caching — return cached responses for similar queries
- Build a dashboard showing routing decisions, costs, and quality metrics over time

## Hints

- The classifier doesn't need to be an LLM — a simple heuristic (query length, keywords, question type) works for v1
- Use `litellm` to avoid writing provider-specific code for each model
- Your eval dataset is the most valuable part of this project — invest time in diverse, representative queries

## Cost Estimate

~$3-5 in API credits. The whole point is to show how routing reduces this.

---

[← Back to Prompt Engineering](../README.md)
