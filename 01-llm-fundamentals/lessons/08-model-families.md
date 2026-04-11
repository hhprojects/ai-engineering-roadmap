# Lesson 8 — Model Families and the Landscape

> **The single sentence version:** There are roughly five families of frontier models you'll actually use in 2026 — Anthropic's Claude, OpenAI's GPT and o-series, Google's Gemini, Meta's Llama, and the Chinese open-weight family (Qwen, DeepSeek) — plus a long tail of specialists; knowing who's good at what is half the battle in production.

By this point in the curriculum, you understand how *a* transformer works. In the real world, you're not choosing whether to use a transformer — you're choosing *which* transformer, from dozens of options with different prices, strengths, and failure modes. This chapter is your map.

All facts below are current as of early 2026. Model lineups turn over every few months; the *shape* of the landscape changes more slowly.

---

## Anthropic — Claude

**The family as of early 2026:** Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5.

| Model | API ID | Context | Input / Output (per MTok) | Best for |
|---|---|---|---|---|
| Claude Opus 4.6 | `claude-opus-4-6` | 1M tokens | $5 / $25 | Frontier intelligence, hard coding and agent work |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 1M tokens | $3 / $15 | The workhorse — best speed/intelligence ratio |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 200k tokens | $1 / $5 | Fast, cheap, near-frontier intelligence |

What defines the Claude family:

- **Instruction following and nuance.** Claude is consistently rated highest at following complex, multi-step instructions and handling ambiguous requests gracefully.
- **Long context that actually works.** Opus and Sonnet both have 1M-token context windows, and Anthropic's own evaluations show them retrieving information reliably across the full window — "needle in a haystack" at 1M tokens is close to 100%.
- **Coding and agent work.** Opus 4.6 is the reference model for agentic coding harnesses (Claude Code, and most third-party coding agents that benchmark well). The model was explicitly post-trained for tool use and multi-turn agent loops.
- **Extended thinking.** All three current Claude models support `thinking` — a reasoning mode that budgets tokens for internal reasoning before producing the final answer. More in Lesson 9.
- **Strong refusal characterization.** Claude is trained with Constitutional AI and is unusually consistent about *which* requests it refuses and *why*. This tends to make it better for production where predictable behavior matters.

When to reach for Claude:

- Anything agent-shaped — coding agents, research agents, multi-turn tool use
- Legal, medical, contract, or any task where nuance and instruction following matter more than raw speed
- Long-document analysis (>100k tokens)
- When you need consistency across calls

When to look elsewhere:

- Image generation (Claude is text and vision in, text out — no image generation)
- Audio (no native audio capabilities)
- When you need the absolute cheapest option at scale (Haiku is competitive but not always the cheapest)

---

## OpenAI — GPT and the o-series

**The family as of early 2026:** GPT-5.4 (the general-purpose flagship), GPT-5.4 mini / nano (smaller variants), and the o-series reasoning models.

OpenAI has two parallel families that do different things:

### GPT-series (general-purpose)

Fast, cheap, good at everything. The GPT-5.4 family is multimodal — text, images, and audio in; text, structured outputs, and function calls out. Native voice mode. The broadest ecosystem — every third-party tool supports OpenAI's chat completions API first.

**Strengths:**
- Extremely low latency on the flagship and mini tiers
- Best-in-class multimodal coverage (voice, vision, image gen via DALL-E/partners)
- Biggest ecosystem of libraries, tools, docs, examples
- The Batch API tier makes large async workloads ~50% cheaper

**Weaknesses:**
- Historically less reliable at complex instruction following than Claude for nuanced tasks
- Content policy is stricter in some domains (adversarial creative writing, security research)
- Per-token cost is often higher than DeepSeek or Qwen for comparable quality

### o-series (reasoning-first)

Entirely separate models optimized for *thinking longer*. They burn tokens on invisible internal reasoning before producing a short final answer. Expensive but qualitatively better at tasks that benefit from deliberation — complex math, scientific reasoning, logic puzzles, dense code refactors.

More on these in Lesson 9.

When to reach for OpenAI:

- Voice agents and real-time audio
- Image generation (DALL-E)
- When you're building on an existing OpenAI-first ecosystem
- When you need the cheapest multimodal option (GPT-5.4 nano is very competitive)
- Reasoning tasks where cost matters less than correctness → o-series

---

## Google — Gemini

**The family as of early 2026:** Gemini 3.1 Pro (the flagship, topping quality benchmarks), Gemini 3 Flash (fast/cheap mid-tier), Gemini 3 Nano (on-device).

Gemini's defining traits:

- **Quality leader on several benchmarks.** Gemini 3.1 Pro has been topping the Artificial Analysis and LMSYS quality leaderboards in 2026, narrowly edging out GPT-5.4 and Claude Opus 4.6 on aggregate scores.
- **Native multimodality.** Gemini was designed from the start as a multimodal model rather than having vision bolted on. It's especially strong on video understanding (the only frontier model that natively ingests video), long-form image reasoning, and cross-modal tasks.
- **Long context at low cost.** Gemini's long-context pricing is aggressively low — processing a 1M-token document is often half the price of the equivalent Claude request.
- **Direct Google Cloud integration.** Vertex AI is the enterprise path.

**Weaknesses:**
- Ecosystem maturity still lags OpenAI and Anthropic for many frameworks
- Instruction following is improving but not yet as reliable as Claude for some domains
- The API surface has been unstable — breaking changes more often than the other majors

When to reach for Gemini:

- Video or complex image understanding
- Very long context at lower cost
- When you're already on Google Cloud and want the native integration
- Benchmark-chasing applications where you want the current "best score"

---

## Meta — Llama (the open-weight incumbent)

**The family as of early 2026:** Llama 4 (the current generation) in sizes ranging from small (~3B) up to the ~400B flagship. The previous Llama 3.1 / 3.2 / 3.3 family is still widely deployed.

Llama is *the* open-weight family. Meta releases the weights publicly, and every inference provider (Groq, Together, Fireworks, Replicate, AWS Bedrock, etc.) hosts them at aggressive prices.

What this means in practice:

- **You can run it yourself.** Llama is the default choice if you want to self-host, fine-tune, or put inference behind your own firewall.
- **Many providers, aggressive pricing.** Because providers compete on the same weights, Llama inference is often the cheapest way to access "close to frontier" quality.
- **Groq is the speed play.** Groq's LPU hardware serves Llama at 200+ tokens/second, making it the fastest option for latency-critical apps.
- **Fine-tuning is ecosystem-rich.** LoRA, QLoRA, Axolotl, Unsloth, torchtune — all the tooling assumes Llama.
- **Quality is behind closed models** but improves every release. Llama 4 is competitive with "previous-generation" closed models (GPT-4 level) rather than matching the current frontier.

When to reach for Llama:

- Self-hosting (compliance, data sovereignty, cost at scale)
- Fine-tuning for specialized tasks
- Latency-critical apps served by Groq
- When you want open weights you can inspect

---

## The Chinese open-weight family — Qwen and DeepSeek

The biggest shift in the 2024-2026 landscape was the rise of competitive open-weight models from China.

### Qwen (Alibaba)

Qwen 2.5 and its successors come in sizes from 0.5B to 72B, with specialized variants for code (Qwen Coder), math, and multilingual tasks. The smaller Qwen models have become the default for on-device and edge deployment. Qwen's multilingual performance — especially on Chinese, Japanese, and Korean — is the best available, including vs. closed models.

### DeepSeek

DeepSeek's claim to fame is shockingly cheap frontier-adjacent quality. DeepSeek V3.2 and the R1 reasoning model are priced at $0.30 per million input tokens — an order of magnitude cheaper than Claude or GPT — while scoring competitively on many benchmarks. DeepSeek-R1 is the leading open-weight reasoning model.

The catch: inference reliability varies by provider, and model behavior is sometimes noticeably different from Western models (different refusal patterns, different cultural defaults). For English-language production use, DeepSeek via a reputable inference provider (Together, Fireworks, OpenRouter) is legitimately competitive on cost per quality.

When to reach for Qwen / DeepSeek:

- Multilingual tasks, especially Chinese and Japanese (Qwen)
- Aggressive cost optimization at scale (DeepSeek)
- Open-weight alternatives where Llama isn't quite enough
- Edge deployment (small Qwen variants)

---

## Specialists and the long tail

Beyond the big families, there's a rich ecosystem of specialized models you should know exist:

- **Mistral (France)** — Mistral Large, Mistral Medium. Competitive quality, strong European data-residency story, open weights for several variants.
- **Cohere** — Command family. Strong at RAG and enterprise search, native multilingual.
- **xAI — Grok** — Frontier-adjacent quality with unusual training data (includes Twitter/X). Grok 4.20 (yes, that's the version number) is competitive on speed.
- **NVIDIA Nemotron** — Nemotron 3 Super, optimized for inference on NVIDIA hardware. Very fast.
- **gpt-oss-120B** — OpenAI's open-weight release; available via many providers at very low cost.
- **Phi (Microsoft)** — Small, efficient models. Phi-3 and successors punch above their weight at ~3-7B sizes.
- **Gemma (Google)** — Open-weight cousins of Gemini. Gemma 3 (270M to 27B) is the default for tiny models on Ollama.
- **Embedding models** — You'll cover these in Module 3 (RAG). The current leaders are OpenAI's text-embedding-3, Cohere Embed v3, and Voyage AI's voyage-3.
- **Whisper (OpenAI)** — speech-to-text. Still the default for most use cases.

---

## Benchmarks, briefly

You'll see models ranked against benchmarks like **MMLU** (general knowledge), **HumanEval** and **SWE-Bench** (coding), **GSM8k** and **MATH** (math), **GPQA** (science), **LiveBench** (a moving-target benchmark that rotates questions to avoid contamination). There are also head-to-head human preference leaderboards — **LMSYS Chatbot Arena** is the most-cited.

Rules for using benchmarks:

- **Never pick a model based on one benchmark.** Models overfit public benchmarks constantly. Look at five or more, including at least one that the model makers didn't train for.
- **Look at performance on *your* task.** A model that's #1 on MMLU might be mediocre at the specific thing you care about. Build a small eval set of your own real prompts (~30-100) and test your shortlist on it. Module 5 (Observability) goes deep on this.
- **Watch the cost axis too.** A model that's 1 point better on a benchmark but 5× more expensive is almost never the right choice. Artificial Analysis has a useful "quality vs. price" scatter.
- **Favor recent, independent benchmarks.** LiveBench, SWE-Bench Verified, and Artificial Analysis update more reliably than older academic benchmarks.

Useful leaderboards to bookmark:

- [LMSYS Chatbot Arena](https://lmarena.ai/) — human preference votes
- [Artificial Analysis](https://artificialanalysis.ai/) — quality vs. price and speed
- [LiveBench](https://livebench.ai/) — contamination-resistant benchmark suite
- [SWE-Bench Verified](https://www.swebench.com/) — realistic software engineering tasks

---

## What to remember from this lesson

- Five families dominate frontier work: **Claude, GPT/o-series, Gemini, Llama, and the Chinese open-weight family (Qwen/DeepSeek)**.
- Claude is the pick for agents, coding, and complex instruction following. GPT owns multimodal and voice. Gemini wins on video and cheap long context. Llama wins when you need open weights. DeepSeek wins on cost per quality.
- Benchmarks are useful directional signals, not truth. Build your own eval set on your actual tasks.
- Prices and lineups turn over every 3-6 months. Re-check pricing pages before committing to a model.
- Know the specialists (Mistral, Cohere, Whisper, Phi, Gemma, embedding models) so you can reach for them when general-purpose models don't fit.

Lesson 13 ("Choosing a Model") turns this survey into an actual decision framework. The next few lessons cover capabilities that cut across families: reasoning, structured outputs, and pricing tricks.

---

## References

- Anthropic, *Claude models overview* (current pricing and capabilities). https://docs.claude.com/en/docs/about-claude/models/overview
- Anthropic, *Introducing the Claude 4 family*. https://www.anthropic.com/news/claude-4
- OpenAI, *Models documentation*. https://platform.openai.com/docs/models
- Google DeepMind, *Gemini models*. https://deepmind.google/technologies/gemini/
- Meta AI, *Llama models*. https://llama.meta.com/
- Alibaba Cloud, *Qwen models*. https://qwenlm.github.io/
- DeepSeek, *DeepSeek models*. https://www.deepseek.com/
- Artificial Analysis, *LLM performance and pricing leaderboard*. https://artificialanalysis.ai/
- LMSYS, *Chatbot Arena leaderboard*. https://lmarena.ai/
- Ollama, *model library*. https://ollama.com/library

---

[← Lesson 7](07-sampling-strategies.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 9: Reasoning Models](09-reasoning-models.md)
