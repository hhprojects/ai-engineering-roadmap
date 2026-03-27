# §1 — LLM Fundamentals

**Goal:** Understand how transformers work, key model families, and the economics of tokens.

## Learning Objectives

- Explain the transformer architecture (attention, embeddings, positional encoding)
- Compare major model families and their trade-offs (GPT, Claude, Llama, Mistral)
- Understand tokenization, context windows, and what drives API costs

---

## 📚 Readings & Resources

| Resource | Type | Why |
|----------|------|-----|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Blog | Jay Alammar's classic — the gold standard visual explainer |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | Interactive | Interactive visual walkthrough — play with attention in your browser |
| [How Transformers Work](https://www.datacamp.com/tutorial/how-transformers-work) | Tutorial | DataCamp deep dive with code examples |
| [DeepLearning.AI: How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) | 🎬 Course | Free short course from Andrew Ng's team |
| [Andrej Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 🎬 YouTube | 2hr code-along, builds a transformer from zero |
| [3Blue1Brown: But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M) | 🎬 YouTube | Beautiful visual intuition for attention and transformers |

---

## 🔨 Projects

| # | Project | Difficulty | Description |
|---|---------|------------|-------------|
| 1 | [Model Comparison Notebook](projects/01-model-comparison.md) | 🟢 Beginner | Call 3+ APIs, compare responses, log costs |
| 2 | [Token Economics Calculator](projects/02-token-economics.md) | 🟡 Intermediate | Tokenize, count, estimate costs across providers |
| 3 | [Mini Transformer](projects/03-mini-transformer.md) | 🟠 Advanced | Build a character-level transformer from scratch |

---

## Key Concepts

After completing this section, you should understand:

- What attention is and why it matters (self-attention, multi-head attention)
- How tokens are created from text (BPE, WordPiece)
- The difference between encoder, decoder, and encoder-decoder architectures
- Context windows and why they have limits
- How model size relates to capability and cost
- The major model providers and their strengths (OpenAI, Anthropic, Meta, Google)
- Why "temperature" and "top-p" affect output randomness
- Basic API patterns: chat completions, system/user/assistant messages

---

[← Foundations](../00-foundations/) | [Home](../README.md) | [Next → Prompt Engineering](../02-prompt-engineering/)
