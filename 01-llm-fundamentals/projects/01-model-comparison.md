# Model Comparison Notebook

🟢 **Beginner**

Your first hands-on encounter with LLM APIs. Call multiple models with identical prompts and see how they differ — in style, accuracy, speed, and cost.

## What You'll Build

A Jupyter notebook that sends the same set of prompts to 3+ model APIs, displays responses side-by-side, and logs token usage, latency, and estimated cost per call.

## What You'll Learn

- Making API calls to OpenAI, Anthropic, and Groq
- Understanding API response structures
- Measuring and comparing model performance
- Working with Jupyter notebooks for experimentation
- Basic cost awareness for LLM usage

## Tech Stack

- Python 3.11+
- `openai` SDK
- `anthropic` SDK
- `groq` SDK
- Jupyter Notebook
- `pandas` for data display

## Requirements

- Call at least 3 different models (e.g., GPT-4o-mini, Claude 3.5 Haiku, Llama 3 via Groq)
- Use at least 5 diverse prompts (creative writing, code, factual Q&A, reasoning, summarization)
- For each call, log: model name, prompt, response, token count (input + output), latency (ms), estimated cost
- Display results in a comparison table
- Include markdown cells with your analysis: which model was best for what?
- Visualize cost vs. quality trade-offs (simple bar charts are fine)

## Stretch Goals

- Add streaming support and compare time-to-first-token
- Include a local model via Ollama for comparison
- Create a scoring rubric and rate each response on a 1-5 scale

## Hints

- Use `time.perf_counter()` around API calls for accurate latency measurement
- Check each provider's pricing page and calculate cost from token counts — don't hardcode
- Start with the cheapest models to iterate on your notebook structure before burning tokens on expensive ones

## Cost Estimate

~$1-2 in API credits for all experiments combined. Use Groq (free) for most iteration.

---

[← Back to LLM Fundamentals](../README.md)
