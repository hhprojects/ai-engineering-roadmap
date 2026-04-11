# 1 — LLM Fundamentals

**Goal:** Build a first-principles understanding of how modern language models work — from tokens and attention to reasoning models, structured outputs, and the economics of running them in production.

This module is a self-contained textbook. You don't need to follow external links to complete it (though they're there if you want to go deeper). Every chapter synthesizes material from the canonical sources, explains it in plain language, and ends with a references section so you can verify or dig in.

Budget: **~15-20 hours** of reading + ~20 hours for the three projects.

---

## How to use this module

1. **Read the lessons in order**, one per sitting. Each chapter takes 15-30 minutes and builds on the previous one.
2. **Do the projects in parallel with the reading** — Project 1 after Lesson 8, Project 2 after Lesson 11, Project 3 after Lesson 7. The sequencing is called out in each project's prerequisites.
3. **Don't skip to Module 2** until you can answer the "what to remember" checklist at the bottom of each lesson without looking back.
4. **Keep a notes file.** Every time something surprises you or contradicts your mental model, write it down. That's the material you'll actually retain.

---

## 📖 Lessons

Read in order. Each lesson ends with a "what to remember" summary and a references section.

| # | Lesson | Focus | Est. time |
|---|---|---|---|
| 1 | [What Is a Language Model?](lessons/01-what-is-a-language-model.md) | Next-token prediction, pre-training vs post-training | 20 min |
| 2 | [Tokenization](lessons/02-tokenization.md) | BPE, WordPiece, SentencePiece, what tokens actually are | 25 min |
| 3 | [Embeddings & Positional Encoding](lessons/03-embeddings-and-positional-encoding.md) | How tokens become vectors and why position matters | 25 min |
| 4 | [The Transformer Architecture](lessons/04-transformer-architecture.md) | Decoder-only blocks, attention + MLP, residuals, norms | 30 min |
| 5 | [Attention](lessons/05-attention.md) | Q, K, V, multi-head, causal masking, GQA, Flash Attention | 35 min |
| 6 | [Training and Inference](lessons/06-training-and-inference.md) | Pre-training, SFT, RLHF, prefill vs decode, KV cache | 30 min |
| 7 | [Sampling Strategies](lessons/07-sampling-strategies.md) | Temperature, top-p, top-k, logprobs, penalties | 25 min |
| 8 | [Model Families and the Landscape](lessons/08-model-families.md) | Claude, GPT, Gemini, Llama, Qwen, DeepSeek, specialists | 30 min |
| 9 | [Reasoning Models](lessons/09-reasoning-models.md) | o-series, extended thinking, DeepSeek R1, when to use | 30 min |
| 10 | [Structured Outputs and Tool Use](lessons/10-structured-outputs-and-tool-use.md) | JSON schemas, strict mode, function calling, parallel tools | 30 min |
| 11 | [Token Economics](lessons/11-token-economics.md) | Pricing, prompt caching, batch APIs, routing, cost optimization | 30 min |
| 12 | [Running Models Locally](lessons/12-running-models-locally.md) | Ollama, llama.cpp, vLLM, quantization, hardware sizing | 30 min |
| 13 | [Choosing a Model](lessons/13-choosing-a-model.md) | Decision tree, evals, benchmarks, routing in production | 25 min |

---

## 🔨 Projects

Three projects, one per difficulty tier. Each is fully spec'd with setup, requirements, rubrics, and common pitfalls — you should be able to complete them without reading anything outside the project file.

| # | Project | Difficulty | Time | Cost | Prereq lessons |
|---|---|---|---|---|---|
| 1 | [Model Comparison Notebook](projects/01-model-comparison.md) | 🟢 Beginner | 3-4 hrs | ~$2 | 1, 2, 7, 8, 11 |
| 2 | [Token Economics Calculator](projects/02-token-economics.md) | 🟡 Intermediate | 5-7 hrs | <$1 | 2, 8, 11 |
| 3 | [Mini Transformer from Scratch](projects/03-mini-transformer.md) | 🟠 Advanced | 8-12 hrs | free | 1-7 |

---

## Supplementary resources

If you want to go deeper than the lessons, these are the canonical sources. Most of the lesson content is synthesized from them.

### Videos
- [Andrej Karpathy — *Let's build GPT from scratch*](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 2hr code-along, required for Project 3
- [3Blue1Brown — *But what is a GPT?*](https://www.youtube.com/watch?v=wjZofJX0v4M) — best visual intuition for attention
- [3Blue1Brown — *Attention in transformers, visually explained*](https://www.youtube.com/watch?v=eMlx5fFNoYc) — companion video on attention

### Interactive
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — click through a live GPT-2
- [LMSYS Chatbot Arena](https://lmarena.ai/) — the human-preference leaderboard
- [Artificial Analysis](https://artificialanalysis.ai/) — current model pricing, quality, and speed

### Reading
- [Jay Alammar — *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) — the classic explainer
- [Anthropic — *Transformer Circuits Thread*](https://transformer-circuits.pub/) — interpretability research
- [Lilian Weng — *Transformer Family*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — comprehensive reference
- [Vaswani et al. — *Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — the original paper; re-read it after Project 3

### Provider documentation
- [Anthropic docs](https://docs.claude.com/) — the most thorough docs in the space
- [OpenAI docs](https://platform.openai.com/docs)
- [Google AI for Developers](https://ai.google.dev/)
- [Hugging Face docs](https://huggingface.co/docs)
- [Ollama library](https://ollama.com/library)

---

## Key concepts — the exit checklist

By the time you finish this module, you should be able to explain all of these in plain language, without looking anything up:

- [ ] What an LLM is, at the simplest level (next-token predictor)
- [ ] How tokenization works and why BPE is the default
- [ ] What embeddings are and how position information gets added
- [ ] The three architecture families and why decoder-only dominates
- [ ] How self-attention works (Q, K, V, softmax, weighted sum)
- [ ] Why multi-head attention exists and what heads specialize in
- [ ] Causal masking and why it makes parallel training possible
- [ ] The difference between prefill and decode and why output tokens cost more
- [ ] What KV caching is and why it matters for long conversations
- [ ] How temperature, top-p, and top-k affect generation
- [ ] What distinguishes Claude, GPT, Gemini, Llama, and DeepSeek
- [ ] How reasoning models differ from chat models and when to use them
- [ ] Structured outputs vs. JSON mode vs. raw text parsing
- [ ] The three or four biggest levers for cutting LLM costs
- [ ] Which hardware can run which model sizes, and what quantization trades off
- [ ] How to choose a model for a new task (the decision tree + eval)

If any of these feel shaky, go back to the corresponding lesson.

---

## A note on staleness

The LLM landscape moves fast. The lessons name specific model versions (Claude Opus 4.6, GPT-5.4, Gemini 3.1, etc.) as of **early 2026**. By the time you read this, those names will have shifted. The *concepts* are stable; the *lineups* turn over every 3-6 months.

When a lesson refers to a specific model:
- Treat the name as an example, not a permanent fact
- Check the provider's current model page before quoting pricing or capabilities
- The broad shape of the landscape (five families, reasoning tier, flagship/mid/mini tiers, open-weight vs closed) is what you're really learning

The [references](#supplementary-resources) section above links to always-current sources. Bookmark them.

---

[← Foundations](../00-foundations/) | [Home](../README.md) | [Next → Prompt Engineering](../02-prompt-engineering/)
