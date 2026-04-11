# Project 1 — Model Comparison Notebook

🟢 **Beginner** · ~3-4 hours · ~$2 in API credits

Your first hands-on encounter with LLM APIs. Call multiple models with the same prompts, see how they differ, log what each one actually costs, and build the mental model that will let you make informed model-selection decisions later.

---

## Prerequisites

- Finished **Lessons 1, 2, 7, 8, 11** (what an LLM is, tokenization, sampling, model families, economics)
- Python 3.11+ installed
- API keys for at least 3 providers (see [Setup](#setup))
- Basic familiarity with Jupyter notebooks

---

## What you'll build

A Jupyter notebook (or Python script) that:

1. Sends the **same set of prompts** to ≥3 model APIs.
2. Displays responses **side-by-side** in a comparison table.
3. Logs **token counts, latency, and estimated cost** for every call.
4. Produces a short **analysis** — which model was best at what, where they disagreed, where the cost-quality trade-off was worth it.
5. Saves the raw results to a JSON or CSV file for your evaluation records.

This is the notebook you'll actually use as a reference every time you're deciding between models for a future project. Treat it like a tool you're building, not a homework assignment.

---

## What you'll learn

- Calling three different LLM APIs and understanding their response structures
- Measuring latency and computing cost from token usage
- Building your first small evaluation set (the habit from Lesson 13)
- Basic data munging with pandas for side-by-side comparison
- Recognizing differences in model style, bias, refusal patterns, and accuracy

---

## Tech stack

- **Python 3.11+**
- `openai` — OpenAI SDK (for GPT)
- `anthropic` — Anthropic SDK (for Claude)
- `groq` — Groq SDK (for Llama on fast hardware)
- `pandas` — tables
- `matplotlib` — simple plots
- `python-dotenv` — loading API keys from a `.env` file
- Jupyter Notebook (`pip install jupyter` or use VS Code's built-in notebook support)

---

## Setup

### 1. Create a project directory

```bash
mkdir model-comparison && cd model-comparison
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install openai anthropic groq pandas matplotlib python-dotenv jupyter
```

### 2. Get API keys

- **OpenAI** — https://platform.openai.com/api-keys (add ~$5 in credits; you'll use <$1)
- **Anthropic** — https://console.anthropic.com/ (same — ~$5 in credits is plenty)
- **Groq** — https://console.groq.com/keys (free tier, generous rate limits — great for iteration)

Put them in a `.env` file:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

Add `.env` to `.gitignore`. Never commit keys.

### 3. Load them in your notebook

```python
from dotenv import load_dotenv
import os
load_dotenv()

# Verify they're loaded
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY"
assert os.getenv("GROQ_API_KEY"), "Missing GROQ_API_KEY"
```

---

## Requirements

### Must have

**Models** — use at least these three (you can swap specific model IDs):

- **OpenAI** — `gpt-4o-mini` or `gpt-5.4-nano` (cheap OpenAI model)
- **Anthropic** — `claude-haiku-4-5` (cheap Claude model)
- **Groq** — `llama-3.3-70b-versatile` (free, fast)

Use the *cheapest* model from each family for iteration. You can rerun against flagships once your notebook works.

**Prompts** — at least 5 diverse prompts across these categories:

1. **Factual Q&A** — "What is the boiling point of water in Kelvin?"
2. **Creative writing** — "Write a haiku about a sleepy cat on a rainy Sunday."
3. **Code generation** — "Write a Python function that returns the nth Fibonacci number using memoization."
4. **Reasoning** — "If I have 3 apples and I give 2 to Alice, then Bob gives me 4, how many do I have?"
5. **Summarization** — "Summarize the plot of Hamlet in exactly three sentences."

Pick at least two more of your own — ideally things you'll actually care about later.

**For each call, log these fields:**

- `prompt_id` (which prompt)
- `model` (which model)
- `response` (the text output)
- `input_tokens`
- `output_tokens`
- `latency_ms` (from request start to response complete)
- `cost_usd` (computed from token counts and the model's price)

**Display requirements:**

- A pandas DataFrame showing all results
- A pivot table or side-by-side view comparing responses for a single prompt across models
- At least one chart: cost vs. latency, or quality-score vs. cost (quality can be your manual 1-5 rating)
- Markdown analysis cells that describe what you observed

### Stretch goals (pick ≥1)

- **Streaming + time-to-first-token.** Use streaming APIs and log how long until the first token arrives, not just the full response.
- **Local model.** Install Ollama (Lesson 12) and add a local Llama or Qwen model to the comparison.
- **Scoring rubric.** Define a 1-5 scale per prompt and score each response. Compute per-model average scores.
- **Repeat runs.** Call each model 3 times per prompt to see variance. How consistent are they?
- **Structured outputs.** Repeat one prompt with OpenAI's structured outputs (Lesson 10). Compare reliability.

---

## Starter scaffold

Here's the outline. Fill in the body.

```python
# ============================================================
# Cell 1 — Imports and setup
# ============================================================
import os
import time
import json
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
from anthropic import Anthropic
from groq import Groq

load_dotenv()

openai_client = OpenAI()
anthropic_client = Anthropic()
groq_client = Groq()


# ============================================================
# Cell 2 — Pricing table (as of your run date — VERIFY before use)
# ============================================================
PRICING = {
    # $ per million tokens, (input, output)
    "gpt-4o-mini":                (0.15,  0.60),
    "claude-haiku-4-5":           (1.00,  5.00),
    "llama-3.3-70b-versatile":    (0.59,  0.79),   # Groq pricing
}

def cost_usd(model, input_tokens, output_tokens):
    in_price, out_price = PRICING[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


# ============================================================
# Cell 3 — Dataclass for results
# ============================================================
@dataclass
class Result:
    prompt_id: str
    model: str
    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


# ============================================================
# Cell 4 — Model callers (one per provider)
# ============================================================
def call_openai(prompt_id: str, prompt: str, model: str) -> Result:
    start = time.perf_counter()
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return Result(
        prompt_id=prompt_id,
        model=model,
        response=resp.choices[0].message.content,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        latency_ms=elapsed_ms,
        cost_usd=cost_usd(model, resp.usage.prompt_tokens, resp.usage.completion_tokens),
    )


def call_anthropic(prompt_id: str, prompt: str, model: str) -> Result:
    start = time.perf_counter()
    resp = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    text = "".join(b.text for b in resp.content if b.type == "text")
    return Result(
        prompt_id=prompt_id,
        model=model,
        response=text,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_ms=elapsed_ms,
        cost_usd=cost_usd(model, resp.usage.input_tokens, resp.usage.output_tokens),
    )


def call_groq(prompt_id: str, prompt: str, model: str) -> Result:
    start = time.perf_counter()
    resp = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return Result(
        prompt_id=prompt_id,
        model=model,
        response=resp.choices[0].message.content,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        latency_ms=elapsed_ms,
        cost_usd=cost_usd(model, resp.usage.prompt_tokens, resp.usage.completion_tokens),
    )


# ============================================================
# Cell 5 — Define prompts and run
# ============================================================
PROMPTS = [
    ("factual",       "What is the boiling point of water in Kelvin?"),
    ("creative",      "Write a haiku about a sleepy cat on a rainy Sunday."),
    ("code",          "Write a Python function that returns the nth Fibonacci number using memoization."),
    ("reasoning",     "If I have 3 apples and I give 2 to Alice, then Bob gives me 4, how many do I have?"),
    ("summarization", "Summarize the plot of Hamlet in exactly three sentences."),
    # Add two more of your own
]

results: list[Result] = []
for prompt_id, prompt in PROMPTS:
    results.append(call_openai(prompt_id, prompt, "gpt-4o-mini"))
    results.append(call_anthropic(prompt_id, prompt, "claude-haiku-4-5"))
    results.append(call_groq(prompt_id, prompt, "llama-3.3-70b-versatile"))

df = pd.DataFrame([asdict(r) for r in results])
df


# ============================================================
# Cell 6 — Side-by-side view
# ============================================================
pivot = df.pivot(index="prompt_id", columns="model", values="response")
pivot


# ============================================================
# Cell 7 — Cost and latency plots
# ============================================================
# Average cost and latency per model
summary = df.groupby("model").agg(
    avg_cost_usd=("cost_usd", "mean"),
    avg_latency_ms=("latency_ms", "mean"),
    total_cost_usd=("cost_usd", "sum"),
)
summary

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(summary["avg_latency_ms"], summary["avg_cost_usd"])
for model, row in summary.iterrows():
    ax.annotate(model, (row["avg_latency_ms"], row["avg_cost_usd"]))
ax.set_xlabel("Average latency (ms)")
ax.set_ylabel("Average cost per call (USD)")
ax.set_title("Cost vs. latency")
plt.show()


# ============================================================
# Cell 8 — Save raw results
# ============================================================
with open("results.json", "w") as f:
    json.dump([asdict(r) for r in results], f, indent=2)


# ============================================================
# Cell 9 — Analysis (markdown cell — fill this in with prose)
# ============================================================
# Write your observations here:
# - Which model was best at creative writing? Why do you think so?
# - Which model was cheapest? Was it worth it?
# - Did any model refuse anything? Did any make a factual error?
# - If you were shipping a support bot tomorrow, which would you pick and why?
```

---

## Evaluation rubric — how to know you're done

Check each box before calling this project complete:

- [ ] Notebook runs end-to-end from a fresh kernel without errors
- [ ] All 3 providers successfully called at least once
- [ ] ≥5 prompts, ≥3 models → ≥15 rows in the results DataFrame
- [ ] Token counts, latency, and cost are all logged per call
- [ ] Pricing is loaded from a dictionary (not hardcoded inside the call functions)
- [ ] Side-by-side pivot view is displayed
- [ ] At least one chart is rendered
- [ ] A markdown analysis cell contains your honest observations, not boilerplate
- [ ] Raw results are saved to disk (`results.json` or `.csv`)
- [ ] Your `.env` file is in `.gitignore` and you've confirmed no keys are committed

---

## Common pitfalls

- **Hardcoding pricing.** Provider prices change. Keep them in one dict near the top so you can update in one place. Always verify the prices against the provider's pricing page before you trust your cost numbers.
- **Measuring latency with `time.time()`.** Use `time.perf_counter()` — it's monotonic and has higher resolution.
- **Forgetting `max_tokens` on Anthropic.** Anthropic requires `max_tokens` on every request. OpenAI doesn't. Omitting it will throw an error. Use a reasonable cap like 1024.
- **Rate limits on fresh accounts.** Groq and OpenAI have tight limits on new accounts. If you get 429 errors, add a `time.sleep(1)` between calls or reduce your prompt count. Tier 1 accounts usually catch up within a few requests.
- **Counting tokens for the wrong provider.** Each provider returns token counts in slightly different fields. Check the SDK docs. Don't trust `tiktoken` to give you correct counts for non-OpenAI models.
- **Comparing models by one prompt.** You need at least 5 diverse prompts to see meaningful differences. Any single prompt will make one model look disproportionately good or bad.
- **Using flagship models for iteration.** Develop with the cheapest models in each family, then rerun with flagships only when your notebook structure is stable.
- **Parsing Anthropic responses the wrong way.** The response is a list of content blocks; you need to filter to `text` blocks and concatenate. See the scaffold above.
- **Not verifying keys are loaded before calling.** Add assertions. A missing key is the #1 source of confusing errors.

---

## Cost estimate

If you stick to the cheap models and run through the full set of prompts:

| Model | Per-call cost | 5 prompts × 3 runs |
|---|---:|---:|
| `gpt-4o-mini` | ~$0.0002 | ~$0.003 |
| `claude-haiku-4-5` | ~$0.001 | ~$0.015 |
| `llama-3.3-70b-versatile` (Groq) | free tier | $0.00 |

**Total: well under $0.10 for the required work.** Add another ~$1-2 if you want to rerun with flagships for comparison. Budget $5 total and you'll have plenty left over.

---

## What to deliver

A directory containing:

```
model-comparison/
├── notebook.ipynb           ← the main deliverable
├── results.json             ← saved results
├── requirements.txt         ← pinned dependencies
├── .env.example             ← template showing what keys are needed (no real keys!)
├── .gitignore               ← including .env
└── README.md                ← 1-page summary of findings
```

The README should include:
- Which 3 models you compared
- Which 5+ prompts you chose
- Your top 3 observations (plain English, honest)
- Which model you'd reach for next time and why

This is the document you'll actually reference later. Write it for your future self.

---

## Going further (after you finish)

- Repeat the exercise with **reasoning models** on a hard math prompt and observe the quality/cost trade-off (Lesson 9).
- Set up a **local model** via Ollama and add it as a fourth column. What's the quality gap? What's the latency gap? (Lesson 12)
- Build a **simple LLM-as-judge** evaluator that scores each response against a rubric — this is the seed of real evaluation infrastructure (Module 5).
- Turn the notebook into a **reusable script**: `python compare.py --prompts prompts.json --models openai,anthropic,groq`. You'll use this every time you evaluate a new model.

---

## References

- OpenAI, *Python SDK documentation*. https://github.com/openai/openai-python
- Anthropic, *Python SDK documentation*. https://github.com/anthropics/anthropic-sdk-python
- Groq, *Python SDK*. https://console.groq.com/docs/libraries
- Artificial Analysis, *Current model pricing*. https://artificialanalysis.ai/
- Anthropic, *Current pricing page*. https://www.anthropic.com/pricing
- OpenAI, *Current pricing page*. https://openai.com/api/pricing/

---

[← Back to LLM Fundamentals](../README.md) | [Next Project → Token Economics Calculator](02-token-economics.md)
