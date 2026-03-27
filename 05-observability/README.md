# §5 — Observability & Evaluation

**Goal:** Instrument, trace, and systematically evaluate LLM applications.

## Learning Objectives

- Add tracing and observability to any LLM application with Langfuse
- Build evaluation datasets and automated eval pipelines
- Implement LLM-as-judge patterns for scalable quality assessment

---

## 📚 Readings & Resources

| Resource | Type | Why |
|----------|------|-----|
| [Langfuse Docs](https://langfuse.com/docs) | Docs | Open-source LLM observability platform — the standard |
| [Langfuse: Evaluating LLM Applications Roadmap](https://langfuse.com/blog/2025-11-12-evals) | Blog | Practical eval methodology from observability to experiments |
| [Langfuse: RAG Observability and Evals](https://langfuse.com/blog/2025-10-28-rag-observability-and-evals) | Blog | Specific to RAG evaluation workflows |
| [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) | Blog | **Must-read** — the why and how of LLM evaluation |
| [Eugene Yan: Evaluation for LLM Applications](https://eugeneyan.com/writing/llm-patterns/) | Blog | Patterns and anti-patterns from production experience |
| [DataCamp: Getting Started with Langfuse](https://www.datacamp.com/tutorial/langfuse) | Tutorial | Hands-on Langfuse tutorial with code |

---

## 🔨 Projects

| # | Project | Difficulty | Description |
|---|---------|------------|-------------|
| 1 | [Add Tracing](projects/01-add-tracing.md) | 🟢 Beginner | Instrument a previous project with Langfuse |
| 2 | [Eval Suite](projects/02-eval-suite.md) | 🟡 Intermediate | Golden dataset + automated quality metrics |
| 3 | [LLM-as-Judge Pipeline](projects/03-llm-judge-pipeline.md) | 🟠 Advanced | Multi-judge evaluation with inter-rater agreement |

---

## Key Concepts

After completing this section, you should understand:

- Why observability is essential (you can't improve what you can't measure)
- Tracing: spans, traces, and how to visualize LLM call chains
- The difference between online (production) and offline (batch) evaluation
- Golden datasets: what they are and how to build them
- Evaluation metrics: correctness, faithfulness, relevance, toxicity
- LLM-as-judge: using one model to evaluate another
- Inter-rater agreement and when to involve humans
- How to integrate evals into CI/CD pipelines

---

[← Agents](../04-agents/) | [Home](../README.md) | [Next → Production](../06-production/)
