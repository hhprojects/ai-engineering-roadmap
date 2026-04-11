# Lesson 1 — What Is a Language Model?

> **The single sentence version:** A modern language model is a function that, given a sequence of text, predicts the probability of whatever comes next — and it gets surprisingly far just by doing that very well.

Before we open the hood and look at transformers, attention, or tokens, you need to carry one mental model everywhere: an LLM is a *next-token predictor*. That's it. Every other capability — answering questions, writing code, translating Chinese, summarizing a PDF, holding a conversation — is built on top of that one mechanical trick.

If this sounds underwhelming, good. Most of the magic comes from *scale* and *training data*, not from some special trick you missed. Feeling like "is that really all?" is the correct starting point.

---

## Next-token prediction, concretely

Imagine you give a model this input:

```
The capital of France is
```

The model doesn't "look up" the answer. It computes, for every possible next token in its vocabulary, a probability. Something like:

```
 Paris       0.91
 the         0.03
 a           0.02
 located     0.01
 ...
```

It picks one (usually the most likely, possibly with some randomness — more on sampling later), appends it to the input, and runs the whole thing again:

```
The capital of France is Paris
```

Now it predicts the token *after* "Paris" — maybe a period, maybe ", which is the largest", maybe ". It is famous for". This loop — predict, append, repeat — is called **autoregressive generation**, and it's how every GPT, Claude, Llama, Gemini, and Qwen model produces text.

Everything a chatbot does — answer a question, refuse a request, write a haiku — is the model pulling tokens one at a time out of a probability distribution it recomputes after every token.

---

## Why this works at all

Why does "predict the next token" generalize to "reason about physics" or "debug Python"? Two reasons:

**1. Compression is understanding.** To predict the next word in a sentence like *"The Riemann hypothesis states that every non-trivial zero of the zeta function lies on the ..."*, the model has to have internalized what the Riemann hypothesis *is*. There's no way to get that continuation right by surface-level pattern matching. The training objective *forces* the model to compress the structure of the world, because that's the only way to lower its loss.

**2. Scale unlocks emergent behavior.** A 10M-parameter model trained on next-token prediction will babble. A 10B-parameter model will hold a conversation. A 100B-parameter model will write passable code and pass bar exams. We don't fully understand *why* these jumps happen at certain scales — but they do, consistently, across every model family.

The consequence is that a simple objective ("predict the next token on a huge pile of internet text") produces a system that, as it gets bigger, looks increasingly like general-purpose intelligence. That's the whole bet of the LLM industry.

---

## Pre-training vs post-training

A raw pre-trained model is not what you talk to in ChatGPT or Claude. It's a text-continuation engine. Give it "Once upon a time" and it will finish a story. Give it "How do I reset my router?" and it might continue with a bullet list — or it might imagine three more user questions and continue with those. It has no concept of "being an assistant."

Modern chat models go through at least two more stages after pre-training:

- **Supervised fine-tuning (SFT)** — the model is shown tens of thousands of examples of "good" assistant responses written by humans, and learns to imitate them.
- **Reinforcement learning from human feedback (RLHF)** or **constitutional AI / RLAIF** — the model generates multiple candidate responses, humans (or another model) rank them, and the model is nudged toward the ones that got ranked higher.

These post-training steps are what turn a raw text predictor into something that refuses harmful requests, follows instructions, formats responses as lists when asked, and says "I don't know" instead of making things up (most of the time).

Whenever you see a "system prompt" in the API, you're talking to a post-trained model. The pre-trained base is still in there underneath — it's just been coached into a particular persona.

---

## The three things you're paying for

Every time you call an LLM API, you're paying for three kinds of compute:

1. **Input processing** — the model reads your prompt and builds an internal representation of it. Linear in the length of the input.
2. **Output generation** — the model produces tokens one at a time, rerunning part of the network for each one. Linear in the length of the output.
3. **Any "thinking" the model does between reading and answering** — for reasoning models (Chapter 9), this can be 10-100× the visible output.

Input tokens are cheap. Output tokens are expensive (usually 4-5× input cost). Reasoning tokens are *more* expensive. Understanding this ratio is the first step to controlling your LLM spend — we'll come back to it in Chapter 11.

---

## What to remember from this lesson

- An LLM is a function that assigns a probability to every possible next token, given the tokens so far.
- Everything the model "does" — answer, reason, refuse — is built on that one loop: predict, append, repeat.
- Pre-training teaches the model language; post-training teaches it to behave like an assistant.
- You pay per token, and output tokens cost more than input tokens.
- The "magic" is scale + training data, not a special trick.

Keep this chapter's mental model in your head when reading every other chapter. When you get confused about why something works, come back to: *it's predicting the next token*.

---

## References

- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*. https://www.youtube.com/watch?v=kCc8FmEb1nY
- 3Blue1Brown, *But what is a GPT? Visual intro to transformers*. https://www.youtube.com/watch?v=wjZofJX0v4M
- Jay Alammar, *The Illustrated Transformer*. https://jalammar.github.io/illustrated-transformer/
- poloclub, *Transformer Explainer — Interactive visualization of GPT-2*. https://poloclub.github.io/transformer-explainer/

---

[← Back to LLM Fundamentals](../README.md) | [Next → Lesson 2: Tokenization](02-tokenization.md)
