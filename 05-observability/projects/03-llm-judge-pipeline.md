# LLM-as-Judge Pipeline

🟠 **Advanced**

One judge can be biased. Multiple judges with agreement analysis? Now you're doing real evaluation. Build a pipeline that uses multiple LLMs as judges and tracks quality trends over time.

## What You'll Build

An evaluation pipeline that scores outputs using multiple judge models (GPT-4, Claude, Llama), computes inter-rater agreement, flags disagreements for human review, and tracks quality metrics over time with a dashboard.

## What You'll Learn

- LLM-as-judge methodology and its limitations
- Multi-judge evaluation and consensus scoring
- Inter-rater agreement metrics (Cohen's kappa)
- Building quality tracking dashboards
- Integrating evaluation into CI/CD pipelines

## Tech Stack

- Python 3.11+
- `openai`, `anthropic`, `groq` SDKs
- `scikit-learn` for agreement metrics
- Langfuse for logging
- `streamlit` or `gradio` for dashboard
- GitHub Actions for CI integration

## Requirements

- Define evaluation criteria with clear rubrics (1-5 scale) for at least 3 dimensions:
  - Correctness
  - Helpfulness
  - Safety / appropriateness
- Implement judge prompts for each dimension (consistent scoring instructions)
- Run the same evaluation through 3 different judge models:
  - GPT-4o (or GPT-4o-mini)
  - Claude 3.5 Sonnet (or Haiku)
  - Llama 3 via Groq
- Compute inter-rater agreement using Cohen's kappa (pairwise between judges)
- Flag items where judges disagree by more than 2 points for human review
- Build a quality dashboard showing:
  - Average scores per dimension over time
  - Agreement rates between judges
  - Score distributions
  - Flagged items needing human review
- Store all evaluations in a structured format (SQLite or JSON)
- Create a CI pipeline (GitHub Actions) that runs evals on every prompt change
- Generate alerts when quality drops below a threshold

## Stretch Goals

- Implement "judge calibration" — give judges scored examples to align their scoring
- Add human evaluation interface for reviewing flagged items
- Build a trend analysis that detects quality degradation patterns

## Hints

- Cohen's kappa is in `sklearn.metrics.cohen_kappa_score` — it accounts for chance agreement, unlike raw accuracy
- Judge prompts need to be specific: "Score 1-5" is vague; "Score 1 if completely wrong, 3 if partially correct, 5 if fully correct with all key points" is better
- Run judges in parallel with `asyncio.gather()` — 3 sequential judge calls is 3x slower than necessary

## Cost Estimate

~$5-10 for a full eval run across 3 judges × 50+ items. Groq (Llama) is free, which reduces costs.

---

[← Back to Observability](../README.md)
