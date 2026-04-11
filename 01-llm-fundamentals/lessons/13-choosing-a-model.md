# Lesson 13 — Choosing a Model

> **The single sentence version:** There is no "best model" — there's a best model *for this task at this budget with these constraints*, and the fastest way to the right answer is a quick decision tree followed by a small, honest evaluation on your real prompts.

This is the capstone lesson of Module 1. You now know how LLMs work (Lessons 1-7), what the major families look like (Lesson 8), and all the modern capabilities (reasoning, structured outputs, economics, local). The question this chapter answers is: *given a new problem tomorrow morning, how do you decide which model to reach for?*

The answer is not "always Opus" and not "always the cheapest." It's a short decision tree, a short eval, and a willingness to revisit the decision every few months as the landscape shifts.

---

## The first question: what does "right" look like?

Before you can choose a model, you need to name what you're optimizing. Every model choice is a trade-off among:

- **Quality** — how often does the model produce the right answer?
- **Latency** — how long until the user sees the response start / finish?
- **Cost** — dollars per call at your expected volume
- **Context length** — does the model need to handle long inputs?
- **Modality** — text only, or vision / audio / video?
- **Deployment constraints** — can the data leave your servers? Do you need offline?
- **Ecosystem** — tool use, structured outputs, fine-tuning support, SDK quality
- **Reliability / SLA** — uptime, refusal patterns, content policy

You cannot optimize all of these simultaneously. You must rank them for *your* task.

A rough exercise: if a magic genie offered you the option to get 10% better quality *or* 50% lower cost *or* 2× faster responses, which would you pick? The honest answer tells you what matters for this workload. Different workloads have different answers.

---

## The decision tree

Here's a pragmatic decision tree you can run through in five minutes. It won't land you on the perfect model, but it'll put you in the right neighborhood.

```
START
│
├── Does this need voice / real-time audio?
│   └── YES → GPT-5.4 (OpenAI Realtime) or Gemini 3 with audio. STOP.
│
├── Does this need video understanding?
│   └── YES → Gemini 3.1 Pro or Gemini 3 Flash. STOP.
│
├── Does this need image generation?
│   └── YES → DALL-E 3 or the current OpenAI image model, or a specialized
│              model (Flux, Imagen). This isn't a "chat LLM" decision. STOP.
│
├── Does the data need to stay on-premises?
│   └── YES → Self-host open weights (Llama, Qwen, DeepSeek) via vLLM or
│              Ollama. Pick the largest one your hardware can run quantized.
│              STOP.
│
├── Is this a complex reasoning / math / code-planning task?
│   └── YES → Reasoning model (Claude Opus 4.6 with thinking, OpenAI o-series,
│              or DeepSeek-R1). Measure the cost per successful task, not
│              per call. STOP.
│
├── Is this a long-document task (>100k tokens of context)?
│   └── YES → Gemini 3.1 Pro (cheap long context) or Claude Sonnet 4.6
│              (1M context with good retrieval). Avoid if small context fits.
│              STOP.
│
├── Is this a high-volume, cost-sensitive workload?
│   ├── Is "good enough" acceptable? → Haiku 4.5, Gemini 3 Flash,
│   │                                   DeepSeek V3.2, or gpt-oss-120B
│   └── Need flagship quality? → Batch API on Sonnet 4.6 or GPT-5.4
│   STOP.
│
├── Is this general-purpose chat / Q&A / writing?
│   └── Claude Sonnet 4.6 or GPT-5.4. Flip a coin or A/B test.
│   STOP.
│
└── When in doubt → Claude Sonnet 4.6.
    It's the workhorse. You'll rarely regret it.
```

This isn't a replacement for an eval — it's a starting point to narrow from "any model" to "probably one of these 2-3." Then you run a quick eval to pick between them.

---

## The cheap-model-first heuristic

A useful default: **start with the cheapest model that might plausibly work, and only escalate when you can point to concrete failures.**

Concretely:

1. Write your prompt.
2. Try it on **Haiku 4.5** / **Gemini 3 Flash** / **GPT-5.4 nano** / **DeepSeek V3.2**. Pick whichever of these is cheapest.
3. Run 20-30 real examples from your use case. How many does it get right?
4. If the hit rate is acceptable → ship it. Stop.
5. If not → try the next tier up (Sonnet 4.6, GPT-5.4 mid, Gemini 3 Pro). Repeat.
6. If still not → try the flagship (Opus 4.6, GPT-5.4 flagship, Gemini 3.1 Pro). Repeat.
7. If still not → reasoning model.
8. If still not → you have a prompt problem, not a model problem. Go back to Module 2 (Prompt Engineering).

Most engineering teams skip this ladder and go straight to the flagship. It works, but they're paying 10-30× more than necessary. Running the ladder from the bottom costs ~30 minutes of your time and usually finds the cheapest acceptable model within 2-3 tiers.

---

## Building a small eval set (the most important habit)

Benchmarks are public. Public benchmarks get trained on. Your task is not a public benchmark.

The single most effective thing you can do for model selection is: **build a small, private evaluation set on your actual task.**

The minimum viable eval set:

- **20-100 real prompts** from your use case. If you're building a customer support bot, scrape real (anonymized) historical tickets. If you're summarizing contracts, grab 30 real contracts.
- **Expected answers or rubrics** for each. "The answer should include X, Y, and Z" or "The category should be one of these five values" or, at minimum, "I know what a good answer looks like."
- **A simple scoring function.** Exact match for classification. Embedding similarity for semantic match. LLM-as-judge (another model grading the output against your rubric) for open-ended responses.

Run your shortlisted models against this eval set. Compute: quality score, cost, latency. Plot them. Look at failure modes by hand — not just summary statistics. The 5% of cases where a model fails are usually more informative than the 95% where it succeeds.

Module 5 (Observability) has a full chapter on evaluation. For model selection, the minimum is enough: a spreadsheet with one row per example, one column per model, scored by whatever criterion matters for your task.

**Budget for the eval:** ~$5-20 in API credits. It's the highest-ROI spending in the entire project. Never skip it.

---

## What about benchmarks?

Public benchmarks are useful for three things and one thing only:

1. **Directional signal.** A model that scores high on MMLU is probably smarter than a model that doesn't. A model with a high HumanEval score probably writes better code.
2. **Contamination red flags.** If a model scores dramatically higher on old benchmarks (HumanEval, MMLU) than on new ones (SWE-Bench Verified, LiveBench), it's probably overfit and you should discount its top-line scores.
3. **Cost-quality scatter plots.** Artificial Analysis publishes quality-vs-price plots. These are genuinely useful for "what's the Pareto frontier?"

The benchmarks I'd actually check in 2026:

- **[LMSYS Chatbot Arena](https://lmarena.ai/)** — human preference votes on open-ended prompts. The best overall "vibes" benchmark.
- **[LiveBench](https://livebench.ai/)** — rotates questions monthly to avoid contamination. One of the most trustworthy numerical benchmarks.
- **[SWE-Bench Verified](https://www.swebench.com/)** — realistic software engineering tasks. The benchmark that matters if you're building coding tools.
- **[GPQA Diamond](https://github.com/idavidrein/gpqa)** — hard science questions. Shows which models can actually reason.
- **[Artificial Analysis](https://artificialanalysis.ai/)** — aggregates the above plus pricing, latency, and speed into a single dashboard.

**Rules for using them:**

- Look at five or more, not one.
- Always look at the date. A "state of the art" claim from 6 months ago is stale.
- Cross-reference with your own eval. If the leaderboard says "Gemini wins" but your eval says Claude wins on *your* task, trust your eval.
- Watch for score inflation — when a model is suspiciously ahead on one benchmark and only average on the others, something is off.

---

## The "replace one model" question

Once you've shipped with a model, you'll eventually ask: *should I switch?* New models drop all the time. Two rules:

1. **Don't swap models without re-running your eval.** Even when the new model is from the same family. Claude Opus 4.6 is not drop-in compatible with Claude Opus 4.5 — the outputs are slightly different in ways your users may notice. Always eval before swapping.
2. **Pin model versions in production.** Use `claude-opus-4-6-20260115` (the dated alias), not `claude-opus-4-6`, and definitely not `claude-latest`. You want predictable behavior. When the provider releases a new snapshot, *you* decide when to adopt it by running your eval.

A good cadence: re-run your eval against the current flagship of each major family **every 3 months**. If someone has jumped ahead on your metric by 5%+ and is still within budget, it's worth considering a swap. Smaller gains usually aren't worth the risk and the churn.

---

## Model routing (for mature apps)

Once your app is big enough, no single model will be ideal for everything. That's when you build a **router** — a small piece of logic that classifies each request and sends it to the right model:

```python
def route_request(request):
    if request.type == "simple_faq":
        return "haiku-4-5"            # cheap, fast
    elif request.type == "complex_reasoning":
        return "opus-4-6-thinking"    # expensive but correct
    elif request.type == "long_document_qa":
        return "gemini-3-flash"       # cheap long context
    elif request.type == "creative_writing":
        return "sonnet-4-6"           # balanced
    else:
        return "sonnet-4-6"           # default
```

The classification step can be done with:
- **A small classifier model** — fine-tuned Phi or Gemma, cheap and fast
- **A tiny LLM call** — Haiku or Gemini Flash with a short "classify this into one of these categories" prompt
- **Heuristics** — length, keywords, presence of code, etc.

Done well, routing can cut your blended cost by 3-5× vs. sending everything to the flagship, with equal or better quality because the cheaper models are genuinely good at the simple stuff.

---

## Worked example: a document Q&A app

Let's walk through a realistic decision. You're building a Q&A app over your company's internal documentation — employees ask questions, the app retrieves relevant docs and answers.

**Requirements:**
- Internal only, moderate volume (1k queries/day)
- Quality matters — wrong answers erode trust
- Documents range from 1 page to 50 pages
- Not cost-sensitive at this volume
- Responses should feel fast

**Decision tree walk:**

1. Voice? No.
2. Video? No.
3. Image generation? No.
4. Data on-premises? Not required, but preferred if viable. Flag for later.
5. Complex reasoning? No — mostly lookup-and-summarize.
6. Long documents? Up to 50 pages ≈ 30k tokens. Fits in 200k context easily.
7. High-volume cost-sensitive? Not really — 1k/day at Sonnet prices is manageable.
8. General Q&A? Yes. → Sonnet 4.6.

**Shortlist:** Claude Sonnet 4.6, GPT-5.4 flagship, Gemini 3 Flash (cheaper alternative), DeepSeek V3.2 (cheapest acceptable).

**Eval:** Pull 50 real questions from employees + their "correct" answers (as validated by subject-matter experts). Run each model on each question with a RAG pipeline. Score with LLM-as-judge against the expected answer.

**Result** (hypothetical, but representative):

| Model | Quality | Cost / query | Avg latency |
|---|---:|---:|---:|
| Claude Sonnet 4.6 | 92% | $0.015 | 1.8s |
| GPT-5.4 | 91% | $0.018 | 1.4s |
| Gemini 3 Flash | 78% | $0.002 | 0.9s |
| DeepSeek V3.2 | 74% | $0.001 | 1.2s |

The cheap options fail too often for the "wrong answer erodes trust" requirement. Sonnet and GPT are effectively tied on quality; Sonnet is slightly cheaper, GPT is slightly faster. Either is a defensible choice. Flip a coin — or pick based on your team's existing SDK familiarity.

**Ship with Sonnet 4.6.** Set up monitoring (Module 5). Plan to re-eval every 3 months. When a new model drops, don't swap until the eval says so.

---

## What to remember from this lesson

- There is no best model — only best-for-this-task-at-this-budget.
- Run the decision tree first, then eval your shortlist on *your* prompts.
- Start with the cheapest model that might work; escalate only on concrete failures.
- Build a private eval set of 20-100 real examples. It's the single highest-ROI activity in model selection.
- Public benchmarks are useful as directional signals, not as the decision itself.
- Pin model versions in production. Don't let providers silently upgrade you.
- Re-eval every 3 months against the current frontier. Swap if the numbers justify it.
- At scale, build a router that sends different request types to different models. The blended cost falls 3-5× with no quality loss.

This closes Module 1. You now understand what a language model is, how it works, which ones exist, how they're priced, and how to pick one. The next module goes deep on the craft of writing prompts that get the most out of the model you chose.

---

## References

- Artificial Analysis, *LLM comparison dashboard*. https://artificialanalysis.ai/
- LMSYS, *Chatbot Arena leaderboard*. https://lmarena.ai/
- LiveBench, *Contamination-resistant benchmark suite*. https://livebench.ai/
- SWE-Bench, *Software engineering benchmark*. https://www.swebench.com/
- Hamel Husain, *Your AI product needs evals*. https://hamel.dev/blog/posts/evals/
- OpenAI Evals, *Framework for evaluating LLMs*. https://github.com/openai/evals
- Anthropic, *Building evals for Claude*. https://docs.claude.com/en/docs/test-and-evaluate/develop-tests

---

[← Lesson 12](12-running-models-locally.md) | [Back to LLM Fundamentals](../README.md) | [Module Complete → Next Module: Prompt Engineering](../../02-prompt-engineering/)
