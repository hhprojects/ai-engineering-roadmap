# Build an Eval Suite

🟡 **Intermediate**

"It seems to work" isn't good enough. Build a proper evaluation suite with a golden dataset and automated metrics — the kind that catches regressions before your users do.

## What You'll Build

An evaluation framework for your RAG project with 50+ question/answer pairs, automated metrics for correctness, faithfulness, and relevance, and integration with Langfuse for tracking results over time.

## What You'll Learn

- Creating golden datasets for evaluation
- Implementing automated eval metrics
- Measuring retrieval quality (hit rate, MRR)
- Measuring generation quality (faithfulness, relevance)
- Running A/B experiments on prompt and retrieval changes
- Tracking evaluation results in Langfuse

## Tech Stack

- Python 3.11+
- Your RAG project from §3
- Langfuse for experiment tracking
- `openai` or `anthropic` SDK (for LLM-based metrics)
- pytest for running eval suites
- `pandas` for results analysis

## Requirements

- Create a golden dataset with 50+ entries, each containing:
  - Question
  - Expected answer (or key facts that should appear)
  - Expected source document(s)
- Implement these retrieval metrics:
  - **Hit rate** — is the correct document in top-k results?
  - **Mean Reciprocal Rank (MRR)** — how high is the correct document ranked?
- Implement these generation metrics:
  - **Correctness** — exact match + fuzzy match against expected answer
  - **Faithfulness** — is the answer grounded in the retrieved context? (LLM-scored)
  - **Relevance** — does the answer actually address the question? (LLM-scored)
- Run the eval suite as a pytest test (one test per metric threshold)
- Log all results to Langfuse as scored observations
- Run at least 2 experiments: compare different chunking strategies or prompts
- Generate a comparison report showing which configuration performs better
- The eval suite should run in under 10 minutes

## Stretch Goals

- Add "answer completeness" as a metric — does the answer cover all expected facts?
- Build a simple HTML report with per-question breakdowns
- Integrate into CI: eval suite runs on every prompt change via GitHub Actions

## Hints

- Start with 20 questions and expand to 50+ once the framework works. Quality over quantity.
- For faithfulness scoring, a simple prompt works: "Is this answer fully supported by the following context? Score 1-5."
- Use `pytest.mark.parametrize` to run each question as a separate test case — makes failures easy to debug

---

[← Back to Observability](../README.md)
