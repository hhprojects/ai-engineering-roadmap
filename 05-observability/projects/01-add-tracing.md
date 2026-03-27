# Add Tracing to a Previous Project

🟢 **Beginner**

You've built AI-powered projects. Now make them observable. Add Langfuse tracing and discover where your time and money actually go.

## What You'll Build

Take your RAG or Agent project from a previous section and integrate Langfuse tracing. Visualize latency per step, token usage, cost breakdown, and full trace chains. Then write a short analysis of what you found.

## What You'll Learn

- Setting up Langfuse (cloud free tier)
- Adding traces and spans to existing code
- Interpreting trace waterfalls and cost breakdowns
- Identifying performance bottlenecks in LLM pipelines
- Using observability data to make optimization decisions

## Tech Stack

- Python 3.11+
- Langfuse Python SDK
- A previous project (RAG from §3 or Agent from §4 recommended)

## Requirements

- Sign up for Langfuse Cloud (free tier: 50k observations/month)
- Integrate the Langfuse SDK into an existing project
- Add tracing to every significant step:
  - LLM calls (model, prompt, response, tokens, latency)
  - Embedding calls
  - Vector database queries
  - Tool executions (if using an agent)
- Use `@observe()` decorators or manual span creation
- Run at least 20 queries through the traced system
- In the Langfuse dashboard, identify:
  - Average end-to-end latency
  - Which step takes the most time
  - Token usage distribution (input vs. output)
  - Cost per query
- Write a short analysis (markdown file) documenting your findings and 2-3 optimization ideas

## Stretch Goals

- Add user feedback collection (thumbs up/down) and log it to Langfuse
- Compare traces across different prompts or chunking strategies
- Set up Langfuse alerts for latency spikes or cost anomalies

## Hints

- Langfuse's `@observe()` decorator is the easiest way to start — just decorate your functions
- The Langfuse dashboard's trace view shows a waterfall diagram — look for the longest bars
- Don't forget to trace embedding calls — they're often a hidden cost in RAG systems

## Cost Estimate

Free — Langfuse cloud free tier covers 50k observations/month.

---

[← Back to Observability](../README.md)
