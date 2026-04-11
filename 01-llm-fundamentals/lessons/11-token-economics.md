# Lesson 11 — Token Economics

> **The single sentence version:** Every LLM bill is the same formula — `input_tokens × input_price + output_tokens × output_price + (maybe) reasoning_tokens × output_price` — and the gap between a clueless invoice and an optimized one is almost always prompt caching, batch APIs, and model selection.

You now know how LLMs work and how to use them. This chapter is about what they cost, how providers price them, and what knobs you can turn to cut your bill by 50-95% without degrading quality. These tricks are not optional — they're the difference between "we're running a production app on LLMs" and "we're going broke running a production app on LLMs."

---

## The basic unit: price per million tokens

Every provider prices the same way: a dollar amount per million tokens (MTok), split between input and output.

Current pricing snapshot (early 2026, round numbers):

| Model | Input / MTok | Output / MTok | Output : Input |
|---|---:|---:|:---:|
| Claude Opus 4.6 | $5 | $25 | 5× |
| Claude Sonnet 4.6 | $3 | $15 | 5× |
| Claude Haiku 4.5 | $1 | $5 | 5× |
| GPT-5.4 (flagship) | ~$5 | ~$20 | 4× |
| GPT-5.4 nano | ~$0.15 | ~$0.60 | 4× |
| Gemini 3 Flash | $0.15 | $0.60 | 4× |
| DeepSeek V3.2 | $0.30 | $1.20 | 4× |
| gpt-oss-120B (via providers) | $0.30 | $0.90 | 3× |

Two things to internalize:

1. **Output is 3-5× more expensive than input.** This is universal — it reflects the compute asymmetry between prefill and decode (Lesson 6). Any cost optimization should focus on reducing output first.
2. **There's a 30× price range for broadly-similar quality.** Haiku is $1/MTok in; DeepSeek is $0.30; Gemini Flash is $0.15. For many workloads, a cheaper model gets you 90% of the quality at 10% of the cost.

---

## Reading your invoice: tokens in, tokens out

Every API response tells you exactly how many tokens it consumed. On Anthropic:

```json
{
  "usage": {
    "input_tokens": 500,
    "output_tokens": 200,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

On OpenAI:

```json
{
  "usage": {
    "prompt_tokens": 500,
    "completion_tokens": 200,
    "prompt_tokens_details": {
      "cached_tokens": 0
    }
  }
}
```

Log these for every request in production. Without this data, you cannot diagnose cost regressions or optimize spend. Build it into your LLM wrapper from day one.

A simple formula for your nightly dashboard:

```
cost_per_request = (input_tokens  × input_price_per_mtok  / 1_000_000)
                 + (output_tokens × output_price_per_mtok / 1_000_000)
                 + (cached_read_tokens  × cache_read_price_per_mtok  / 1_000_000)
                 + (cache_write_tokens  × cache_write_price_per_mtok / 1_000_000)
```

Store this per-request in your telemetry. Module 5 (Observability) has much more.

---

## Trick 1: prompt caching

The single biggest lever in modern LLM pricing. If you're sending the same prompt prefix across multiple requests — same system prompt, same tool definitions, same long document, same conversation history — you can have the provider **cache** that prefix and charge you a fraction of the normal rate on subsequent reads.

### How Anthropic's prompt caching works

Anthropic was first to ship this as a real feature, and their implementation is the clearest one to study.

You add a `cache_control` marker to a piece of content in your request:

```python
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are analyzing legal documents...",
            "cache_control": {"type": "ephemeral"}
        },
        {
            "type": "text",
            "text": "[50-page legal agreement]",
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": "What are the payment terms?"}]
)
```

What happens:

1. **First request (cache write).** Anthropic processes the full prompt and caches the prefix up to the last `cache_control` marker. You pay a **25% premium** on those tokens (1.25× the normal input price).
2. **Subsequent request within 5 minutes (cache hit).** Anthropic skips re-processing the cached prefix and charges you **0.1× the normal input price** for it — a 90% discount.
3. **The cache persists for 5 minutes by default.** Every cache hit resets the timer (sliding window). If 5 minutes elapse without a hit, the cache entry is evicted.
4. **1-hour cache is available** at a 2× write premium instead of 1.25×, for workloads that access the cache less frequently.

### The cost math

Concrete example. You have a 50,000-token system prompt + document context, and you make 100 requests against it in a 5-minute window. Each request also has ~500 tokens of unique query and ~300 tokens of output.

**Without caching** (Claude Sonnet 4.6 at $3 input, $15 output):

```
Per request: 50,500 × $3 / 1M  +  300 × $15 / 1M
           = $0.1515 + $0.0045
           = $0.156
Total (100 requests): $15.60
```

**With caching:**

```
First request:  50,000 × ($3 × 1.25) / 1M + 500 × $3 / 1M + 300 × $15 / 1M
             = $0.1875 + $0.0015 + $0.0045 = $0.1935

Subsequent 99: 50,000 × ($3 × 0.1) / 1M + 500 × $3 / 1M + 300 × $15 / 1M
             = $0.015 + $0.0015 + $0.0045 = $0.021 each
             × 99 = $2.08

Total: $0.1935 + $2.08 = $2.27
```

**Savings: $15.60 → $2.27, ~85% off.** The break-even is after ~2 cache hits — literally the second request pays back the cache write premium.

### Where to put cache breakpoints

Order matters. Anthropic's cache uses **exact prefix matching** — it can only reuse content that's identical up to the cache breakpoint. So put **static content first**, **dynamic content last**:

```
[System instructions]          ← cache breakpoint here (most static)
[Tool definitions]             ← or here
[Large document context]       ← or here (frequently queried, changes slowly)
[Conversation history so far]  ← grows with each turn
[Latest user message]          ← always different (no cache here)
```

Tool definitions, system prompts, and knowledge base context are great cache targets. User-specific messages and timestamps are not.

### OpenAI's automatic caching

OpenAI caches prompts automatically — you don't have to mark anything. As of early 2026, OpenAI applies a ~50% discount on cached prompt tokens (less aggressive than Anthropic's 90%, but no configuration needed). Cached tokens show up as `cached_tokens` in the usage response.

Google Gemini has a similar automatic caching approach for long contexts with significant discounts.

The practical upshot: on any provider, if you're not structuring your prompts to take advantage of caching, you're leaving substantial money on the table.

---

## Trick 2: the Batch API

Almost every major provider offers a **batch API** for workloads you can process asynchronously:

- **OpenAI Batch API** — 50% discount on input *and* output, up to 24-hour turnaround
- **Anthropic Message Batches API** — 50% discount, up to 24-hour turnaround
- **Google Gemini Batch mode** — similar discount, similar turnaround

You submit a file with thousands of requests, the provider processes them whenever compute is available, and you download results when done. For workloads like overnight evaluation runs, bulk content generation, research analysis, or backfilling a database — this is a 2× cost win for zero engineering effort.

**When to use batch:**
- Nightly eval runs on a test set of hundreds of prompts
- Bulk summarization of archived content
- Offline fine-tuning data generation
- Any workload where "results by tomorrow morning" is fine

**When not to use batch:**
- Any interactive use
- Anything where you need results within minutes

---

## Trick 3: model selection for the right tier

The laziest cost optimization is also the biggest one for most apps: are you using the most expensive model when a cheaper one would work?

**The tiering strategy.** Every major family has a small, medium, and large model at 1×, 3×, and 10× the price. Route easy tasks to the small one, hard tasks to the big one.

Example flow for a customer support app:

```python
def answer_question(question: str, history: list):
    classification = classify_intent(question, model="haiku-4.5")    # small + fast
    if classification == "simple_faq":
        return answer_from_kb(question, model="haiku-4.5")           # small
    elif classification == "account_issue":
        return answer_with_tools(question, model="sonnet-4.6")       # medium
    elif classification == "complex_dispute":
        return answer_with_reasoning(question, model="opus-4.6")     # large
```

In this pattern, ~80% of requests hit the cheapest model, ~15% hit the middle, ~5% hit the top. The blended cost is usually 3-5× lower than routing everything to the top model, with the same or better quality (because the small model is genuinely good at the simple things).

**The route-then-respond pattern.** Use a tiny model (Haiku, Gemini Flash, DeepSeek) to decide *what kind* of request this is, then call the appropriate larger model. The router call is cheap enough to be essentially free at the scale of the flagship call.

---

## Trick 4: shrinking your prompts

Every token you don't send is a token you don't pay for. Basic hygiene:

- **Don't dump entire documents when a summary works.** If you need "the relevant section of a 200-page PDF," use retrieval (Module 3) instead of the whole document.
- **Don't list all tools when only some are relevant.** Tool definitions count against your input. Route the request first, then call the model with only the relevant tool subset.
- **Don't keep unbounded conversation history.** After 10 turns, summarize the first 8 turns into a short recap and drop the verbatim history. The model loses nothing meaningful and saves thousands of tokens per turn.
- **Don't send verbose system prompts when a short one works.** The first draft of any system prompt is 2× longer than necessary. Trim it after you've validated it works.
- **Strip whitespace and trailing commentary from tool results.** Returning `{"result": "ok"}` costs ~5 tokens. Returning a 200-line JSON debug dump costs ~500. The model doesn't need most of it.

---

## Trick 5: output length controls

Output tokens cost 3-5× more than input tokens. Your highest-leverage lever is keeping output short.

- **Use `max_tokens` as a hard cap.** You always know roughly how long a good response should be. Set the cap at 1.5× that.
- **Ask for the answer first.** Prompt the model to give its conclusion in the first paragraph, then optionally elaborate. If your app only needs the first line, you can truncate early.
- **Use structured outputs to constrain length.** A schema with `"maxLength": 200` on each field literally prevents the model from rambling.
- **For reasoning models, control the thinking budget.** Don't set `budget_tokens=20000` if you've verified the task works fine with 5000.

---

## Trick 6: provider arbitrage on open models

If you're using Llama, Qwen, DeepSeek, or any open-weight model, you're not locked to one provider. The same model is available from:

- **Groq** — fastest (200+ tok/s)
- **Together AI** — broad selection, competitive pricing
- **Fireworks AI** — strong uptime, good for production
- **Replicate** — easy integration, per-second billing
- **DeepInfra** — aggressively cheap
- **OpenRouter** — aggregator that routes to whichever provider has capacity
- **Self-hosted** (vLLM, llama.cpp, TGI) — fixed cost per GPU hour

Prices vary 2-5× for the same model across providers. Use OpenRouter for experimentation, move to a specific provider for production once you've validated the one that matches your latency and reliability needs.

---

## A real optimization example

Suppose you have a document Q&A app. Initial implementation:

- Every query sends the full 80k-token document as context
- Uses Claude Opus 4.6 for all queries
- ~1000 queries per day
- Response length: ~500 tokens average

**Baseline cost per query:**
```
80500 × $5/1M  +  500 × $25/1M  =  $0.4025 + $0.0125  =  $0.415
Daily: $415   Monthly: $12,450
```

**After optimizations:**

1. **Prompt caching on the document** (1-hour TTL because queries cluster in working hours): reduces document cost from $0.40 → $0.04 per query after first. Saves ~$12,000/mo.
2. **Route to Sonnet 4.6 by default, escalate to Opus only on hard queries** (~10% of traffic): reduces per-query cost by ~60% on the 90% of traffic.
3. **Retrieval** (Module 3): instead of the full 80k document, send only the top 3 relevant chunks (~3k tokens total). Document context drops 25×. This also makes caching less valuable but not moot.
4. **Batch overnight eval runs**: the 200 daily "self-evaluation" queries run in batch at half price.

**Optimized cost:**
- Routed mix: ~$0.020 per Sonnet query, ~$0.08 per Opus query
- 90% × $0.020 + 10% × $0.08 = $0.026 per query (cached reads)
- Daily: $26   Monthly: $780

**Savings: $12,450 → $780, a 94% reduction.** None of the optimizations individually required a rewrite — they're cumulative wins from understanding the pricing surface.

This is not a contrived example. This *is* production LLM cost optimization. The gap between "we're spending $12k/mo" and "we're spending $780/mo" is entirely in how you structure the calls, not how smart the model is.

---

## Common pitfalls

- **Not logging token usage.** If you don't track it per-request, you can't optimize it. Build the wrapper on day one.
- **Benchmarking without cost.** A model that's 2 points better on a benchmark but 5× more expensive is rarely worth it. Track quality-per-dollar.
- **Caching dynamic content.** Anthropic's cache uses exact prefix matching — caching a prompt with a timestamp or user ID in it is pointless. Move dynamic fields to the end.
- **Forgetting about cache eviction.** A 5-minute cache is useless if your requests come in every 10 minutes. Use the 1-hour cache or batch your requests into bursts.
- **Over-optimizing early.** Don't spend a week building an optimal caching strategy for an app with 10 users/day. First prove the app works; then optimize when the bill gets big enough to notice.
- **Ignoring the output cost.** Most engineers instinctively worry about input (because it's where the prompt is). But output is 3-5× more expensive per token. Shortening a verbose response from 800 tokens to 300 is a bigger win than trimming 500 tokens of input.

---

## What to remember from this lesson

- Every LLM bill: `input × in_price + output × out_price + (reasoning × out_price)`. Track usage per request or you can't optimize.
- Output tokens are 3-5× more expensive than input. Optimize output first.
- **Prompt caching** is the biggest lever — up to 90% off on repeated prefixes. Put static content first, dynamic last.
- **Batch APIs** give you 50% off for async workloads. Use them for everything that isn't interactive.
- **Model tiering** (cheap → medium → flagship) cuts the blended cost of a multi-type workload by 3-5×.
- **Retrieval and prompt trimming** kill long-tail waste. Don't dump full documents when chunks work.
- On open-weight models, shop providers — there's 2-5× price spread for the same weights.
- A production app without these optimizations is paying 5-20× what an optimized version pays.

---

## References

- Anthropic, *Prompt caching*. https://docs.claude.com/en/docs/build-with-claude/prompt-caching
- Anthropic, *Message Batches API*. https://docs.claude.com/en/docs/build-with-claude/batch-processing
- OpenAI, *Prompt caching*. https://platform.openai.com/docs/guides/prompt-caching
- OpenAI, *Batch API*. https://platform.openai.com/docs/guides/batch
- Google, *Gemini context caching*. https://ai.google.dev/gemini-api/docs/caching
- Artificial Analysis, *LLM price and performance comparison*. https://artificialanalysis.ai/
- Anthropic, *Pricing page*. https://www.anthropic.com/pricing

---

[← Lesson 10](10-structured-outputs-and-tool-use.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 12: Running Models Locally](12-running-models-locally.md)
