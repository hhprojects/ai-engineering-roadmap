# Lesson 9 — Reasoning Models

> **The single sentence version:** Reasoning models are LLMs that are trained and deployed to spend a large, variable number of invisible "thinking" tokens before producing their visible answer — paying more compute for better answers on problems where deliberation actually helps.

In 2024-2025, a new model category emerged that isn't just "a bigger GPT." OpenAI called them the **o-series**. Anthropic built **extended thinking** (sometimes called **adaptive thinking**) into Claude. DeepSeek released **R1** as an open-weight alternative. Google's Gemini has **Deep Think**. They all rest on the same core idea: for some problems, the bottleneck isn't model size — it's whether the model is allowed to *deliberate*.

This chapter explains what reasoning models actually do, how they differ from normal LLMs, when they're worth the extra cost, and how to use them without surprising yourself at the end of the month.

---

## The idea in one paragraph

A normal LLM reads your prompt, does a single forward pass, and starts producing output tokens. Its "thinking" all happens during that forward pass — once, in parallel, with a fixed amount of compute per input token. A reasoning model, by contrast, is trained to produce a long internal chain of thought *before* it begins its visible response. That chain of thought is itself generated one token at a time, following the same autoregressive mechanism as any other LLM output — but the user usually doesn't see it, and it's billed separately as **reasoning tokens** or **thinking tokens**.

The result: for tasks where deliberation helps — multi-step math, complex logic, careful code debugging, scientific reasoning — a reasoning model can dramatically outperform a non-reasoning model of similar size, because it's effectively allowed to "think for longer" on demand.

---

## How they're built

The shared recipe (as best we know from the public literature) has three ingredients:

1. **Start with a strong base model** — a pre-trained + SFT'd LLM, just like Claude or GPT.
2. **Train it to produce long chains of thought** — often using reinforcement learning where the reward is correctness on verifiable problems (math, code that passes tests, logic puzzles). This is the critical piece: instead of just imitating human-written reasoning (supervised fine-tuning on CoT data), the model learns through trial and error which reasoning strategies actually lead to correct answers.
3. **Teach it when to think longer vs. shorter** — either through a user-facing parameter (`reasoning_effort`, `budget_tokens`) or through adaptive post-training (the model decides for itself how much to think based on the problem).

The DeepSeek-R1 paper (January 2025) is the most detailed public account of this process — it's worth reading if you want to understand how the RL loop works.

The key insight from that research: letting the model explore freely, with only an outcome-based reward signal, produces much more effective reasoning than training it on human-written chains of thought. The model finds strategies humans wouldn't have written down.

---

## OpenAI's o-series

The first widely-available reasoning models. As of early 2026, the current o-series includes models like **o3** and **o3-mini** (names will have changed by the time you read this — check the current OpenAI docs).

How to use them:

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="o3-mini",
    reasoning_effort="medium",  # "low", "medium", "high"
    messages=[
        {"role": "user", "content": "Prove that there are infinitely many primes."}
    ]
)
```

Key facts about o-series models:

- **You pay for reasoning tokens.** The model generates invisible "thinking" tokens before its visible response, and you're billed for them at the same rate as output tokens. A task with 500 visible output tokens might generate 5000-10000 reasoning tokens.
- **`reasoning_effort` controls how much the model deliberates.** Low is cheap and fast; high is expensive and slow but gets better answers on hard problems. OpenAI's own benchmarks show large quality gains between low and high on math and coding tasks, diminishing on simpler tasks.
- **Reasoning tokens are hidden.** You don't get to see them in the API response. This was controversial when o1 launched — Simon Willison wrote about feeling uneasy about paying for tokens you can't inspect.
- **No system prompts, no streaming, limited tool use (historically).** Early o1 models stripped out many features to simplify training. Later o-series models have been adding these back, but the reasoning models are generally more feature-limited than GPT.
- **Slow.** A single o-series call can take 30 seconds to several minutes. Don't put one in a user-facing loop where a human is waiting.

When to reach for o-series:

- Complex math, scientific reasoning, formal logic
- Code that requires careful planning (not just "write me a function," but "refactor this module to change its invariant")
- Problems where you've seen GPT-5 produce subtly wrong answers that deeper thinking might fix
- Batch, async workloads where latency doesn't matter

When not to:

- Interactive chat
- Simple Q&A where the answer is easy
- Anything cost-sensitive at high volume
- Tasks GPT-5 already gets right reliably

---

## Anthropic's extended thinking / adaptive thinking

Claude took a different path. Instead of releasing a separate reasoning-first model family, Anthropic built **extended thinking** into Claude directly. Every current Claude model (Opus 4.6, Sonnet 4.6, Haiku 4.5) can be asked to think before responding.

The modern Claude API uses **adaptive thinking** on Opus 4.6 and Sonnet 4.6 — the model decides for itself how many thinking tokens to spend based on the problem. Older models (Sonnet 4.5, Opus 4.1) expose a manual `budget_tokens` parameter.

### Adaptive thinking (Opus 4.6, Sonnet 4.6)

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled"},  # adaptive — no manual budget
    messages=[{"role": "user", "content": "Prove that there are infinitely many primes of the form 4n+3."}]
)
```

### Manual budget (Claude 4.5 and earlier)

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "..."}]
)
```

The response comes back with two kinds of content blocks:

```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me work through this step by step. First, I note that...",
      "signature": "WaUjzky..."
    },
    {
      "type": "text",
      "text": "Yes — there are infinitely many primes of the form 4n+3..."
    }
  ]
}
```

What's different about Claude's approach:

- **Every Claude model can do it.** You don't pick a separate reasoning model; you enable thinking on whichever Claude you were going to use. This makes it much easier to A/B test "does thinking help here?"
- **Thinking integrates with tool use.** You can give a Claude with extended thinking a set of tools, and it'll think, call a tool, see the result, think again, call another tool, and so on. This is how Claude Code works under the hood. If you do this, you must preserve thinking blocks across turns — passing them back into the next API call so the reasoning chain stays consistent.
- **You can see a summarized version of the thinking.** By default, Claude 4 returns a "summarized" thinking block rather than the raw chain of thought. You're billed for the full thinking tokens regardless.
- **You can hide thinking with `display: "omitted"`** to reduce latency for streaming. The thinking still happens and still costs money; you just don't wait to see it.

When to use extended thinking on Claude:

- Any task that benefits from deliberation — the same list as o-series
- Agents doing multi-step planning
- Complex coding tasks inside a Claude Code-style loop
- Situations where you want deliberation *and* tool use in the same call (Claude integrates these; o-series is more restricted)

---

## DeepSeek R1 (and the open-weight reasoning world)

**DeepSeek R1** was the first open-weight reasoning model that genuinely competed with the o-series on hard benchmarks. DeepSeek published the weights, the training methodology, and detailed ablations — a significant moment for open research.

Key properties:

- **Open weights.** You can run it yourself (the full model is huge; distilled versions fit on one GPU).
- **Visible thinking.** Unlike OpenAI's o-series, DeepSeek exposes the raw chain of thought. This makes it much easier to debug "why did the model conclude X?"
- **Aggressive pricing via providers.** On Together, Fireworks, or DeepSeek's own API, R1 is often 10-50× cheaper than OpenAI's reasoning models for comparable quality on math and code.
- **Less polish.** DeepSeek's refusal behavior and edge-case handling feel less production-ready than the closed models. For many applications this is fine; for others it's a dealbreaker.

**Distilled versions** (e.g., `deepseek-r1-distill-llama-70b`) train a Llama model to imitate R1's reasoning traces. These are smaller, faster, and much cheaper to run locally — a great option if you want reasoning capabilities on your own hardware.

---

## The economics: why reasoning is expensive

A normal Claude Sonnet call:

```
Input:  500 tokens × $3/MTok  = $0.0015
Output: 400 tokens × $15/MTok = $0.006
Total:                          $0.0075
```

The same call with extended thinking and an 8000-token thinking budget:

```
Input:     500 tokens × $3/MTok  = $0.0015
Thinking: 7000 tokens × $15/MTok = $0.105   ← this is the new line item
Output:    400 tokens × $15/MTok = $0.006
Total:                             $0.113
```

That's a ~15× cost increase for one request. Reasoning models are a premium tier — use them where they matter, not by default.

**Pricing traps to watch for:**

- Reasoning tokens are billed at the **output** rate, not input — even though you don't see them. On most APIs, that's 4-5× the input rate.
- With `display: "omitted"` or summarized thinking on Claude, you're still billed for the *full* thinking, not the summary.
- Reasoning effort / thinking budget is a *maximum*, not a target. The model may use much less on easy problems and save you money.
- Batching reasoning workloads through the Batch API (OpenAI and Anthropic both offer ~50% off for async) is the standard cost optimization.

---

## When to use reasoning models (the decision framework)

Before you reach for a reasoning model, run this check:

1. **Is the task easy for GPT-5 / Claude Sonnet?** If yes, you don't need reasoning. Pay the normal rate.
2. **Does the task benefit from step-by-step deliberation?** Math, code planning, logic, formal proofs, careful document analysis. If yes, reasoning helps. If the task is vibes-based (creative writing, friendly chat), it won't.
3. **Can you tolerate 30s-5min latency?** Reasoning is slow. If you're in an interactive loop, probably no. If you're running overnight, go for it.
4. **Is the cost justified?** Estimate the per-task cost at your expected thinking budget. If it's 15× more than the non-reasoning baseline, is the quality gain worth that?
5. **Have you tried prompting non-reasoning models to "think step by step"?** Chain-of-thought prompting on a normal model gets you some of the benefit for free. If plain CoT is enough, don't reach for reasoning.

A useful heuristic from production experience: reasoning models shine when the problem has a **verifiable correct answer** (math, code that must pass tests, logic puzzles). They're less valuable for open-ended creative tasks, because "reasoning" doesn't meaningfully help you write a better poem.

---

## Common pitfalls

- **Treating reasoning models as "GPT but smarter."** They're not. They're a different tool. They trade latency and cost for deliberation, and the trade only pays off on problems that need it.
- **Forgetting to preserve thinking blocks across tool-use turns.** On Claude, if you drop the thinking block from the previous turn when making the next call, reasoning continuity breaks and tool-use gets worse. Always pass the full thinking block back.
- **Blowing through the thinking budget on simple tasks.** If you set `budget_tokens=20000` and ask "what's 2+2?", the model still may generate several thousand thinking tokens trying to second-guess itself. Use small budgets for easy problems or let adaptive thinking decide.
- **Using o-series in a streaming chat UI.** The "thinking" phase can take 30 seconds to several minutes before any visible output appears. Users will think your app is broken.
- **Assuming visible reasoning means honest reasoning.** The published chain of thought is not necessarily how the model "actually" computed the answer. It's just another sequence of tokens the model generated. Treat it as a useful hint, not as ground truth about model internals.

---

## What to remember from this lesson

- Reasoning models spend a variable number of invisible "thinking" tokens before producing their final answer.
- OpenAI's o-series is a separate model family. Anthropic's extended thinking is built into every Claude. DeepSeek R1 is the open-weight leader.
- Reasoning tokens are billed at the output rate and can dominate the cost of a single request.
- Reasoning helps on math, code planning, logic, and formal reasoning. It doesn't help much on creative or vibes-based tasks.
- Use adaptive thinking (or small manual budgets) by default. Only crank the budget for hard problems.
- Plain chain-of-thought prompting gets you some of the benefit for free — try that first on a normal model.
- Always verify: if your task doesn't verifiably improve with a reasoning model vs. a normal one, don't pay the premium.

---

## References

- Anthropic, *Extended thinking documentation*. https://docs.claude.com/en/docs/build-with-claude/extended-thinking
- Anthropic, *Claude models overview*. https://docs.claude.com/en/docs/about-claude/models/overview
- OpenAI, *Reasoning models overview*. https://platform.openai.com/docs/guides/reasoning
- OpenAI, *Learning to reason with LLMs* (o1 announcement). https://openai.com/index/learning-to-reason-with-llms/
- Simon Willison, *Notes on o1*. https://simonwillison.net/2024/Sep/12/openai-o1/
- DeepSeek, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. https://arxiv.org/abs/2501.12948
- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. https://arxiv.org/abs/2201.11903

---

[← Lesson 8](08-model-families.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 10: Structured Outputs](10-structured-outputs-and-tool-use.md)
