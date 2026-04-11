# Lesson 1 — What Is Prompt Engineering?

> **The single sentence version:** Prompt engineering is the craft of writing inputs to language models that reliably produce the outputs you want — it's half psychology, half API design, and it's the single highest-leverage skill for getting useful work out of LLMs.

You just spent Module 1 learning how LLMs work under the hood. This module is about the surface — the actual words you type into the prompt box — and why those words matter more than you'd think. By the end, you'll have a repertoire of techniques that can turn a mediocre model into a useful one, and you'll know *when* to reach for each.

---

## Why prompts matter at all

Recall from Module 1 that a language model is a next-token predictor that was trained on a huge pile of text and then post-trained (SFT + RLHF) to behave like an assistant. Two consequences of that training loop drive everything in this module:

1. **The model learned a distribution over "what kinds of responses follow what kinds of prompts."** When you write a prompt, you're not giving the model instructions in the way you'd give them to a human employee. You're activating a particular region of its learned distribution. A prompt that looks like a Stack Overflow question pulls the model toward "helpful technical answer." A prompt that looks like a children's story pulls it toward "more story." The model is deeply influenced by the *style* and *shape* of your input, not just its literal meaning.
2. **Post-training taught the model that certain prompt patterns are rewarded.** When annotators ranked responses during RLHF, they were ranking them on patterns: structured answers, numbered steps, "let me think step by step" style reasoning, polite refusals. Your prompt can evoke those trained behaviors — or fail to. A prompt like "please explain step by step" is not just polite; it triggers a learned reasoning mode.

The upshot: small changes to how you phrase things can produce large changes in quality. A well-phrased prompt to GPT-4 Nano can beat a poorly-phrased prompt to Claude Opus 4.6. Prompt engineering isn't optional — it's the layer where the model's latent capability meets your specific task.

---

## What prompt engineering is (and isn't)

Prompt engineering is the craft of:

- Picking the right **structure** for a prompt — where to put instructions, context, examples, and the query
- Choosing the right **examples** (or none) to steer the model toward the pattern you want
- Giving the model room to **think** before committing to an answer
- Controlling the **output format** so your downstream code can parse the result
- Designing prompts that are **robust** to noisy, hostile, or unusual inputs
- **Evaluating** prompts against real data and iterating systematically, not vibes-first
- **Versioning** prompts like code, so you can roll back when a change regresses quality

Prompt engineering is *not*:

- A substitute for picking a better model. If the task is beyond the model's capability, no prompt rescues it.
- A substitute for evaluation. You cannot tell by reading your prompt whether it's good. You have to test it.
- A substitute for fine-tuning when you genuinely need the model to learn new behaviors. Prompting can't teach facts the model doesn't already know well.
- A permanent trick. Prompts that worked on GPT-3.5 may be unnecessary or even harmful on GPT-5. Techniques age. The *instinct* of "how do I think about shaping model behavior?" is what you're building.

---

## The spectrum: from "please" to DSPy

Prompt engineering spans a wide spectrum of effort and formality. Knowing where you are on that spectrum tells you which techniques are worth your time.

### Level 1 — Direct prompting

You type what you want and the model answers. No examples, no structure, no chain-of-thought.

```
Summarize this article in three bullet points: [article]
```

This works surprisingly often on modern models. Start here. If it works, don't over-engineer.

### Level 2 — Structured prompting

You add explicit instructions, a defined output format, maybe a role or context block. Same API call, more intention.

```
You are a financial analyst. Read the article below and produce:

1. A 1-sentence summary
2. The top 3 risks mentioned
3. The author's stated position (bullish / bearish / neutral)

Return the result in JSON with keys: summary, risks, position.

Article:
[article]
```

This is where ~80% of production prompt engineering lives. You'll spend most of this module on the techniques that make structured prompts robust.

### Level 3 — Few-shot and chain-of-thought

You include examples (few-shot) or you nudge the model to think out loud before answering (chain-of-thought). Both techniques are covered in depth in Lessons 4 and 5. Both can turn a failing prompt into a working one.

### Level 4 — Prompt chaining / multi-step pipelines

You split a complex task into multiple LLM calls, passing intermediate results between them. "Draft → Critique → Revise" is the classic pattern. Useful when a single prompt can't fit everything, or when you need inspectable intermediate state.

### Level 5 — Prompt programming frameworks (DSPy, BAML)

Instead of writing prompts by hand, you write a **program** that describes what you want and lets a framework generate and optimize the prompts for you. DSPy is the most prominent of these. We cover them in Lesson 13.

You don't need to start at Level 5. Most of the value comes from doing Level 2 well. But it's useful to know the ladder exists so you're not surprised when a team you join is using any particular rung.

---

## The golden rule of prompting

Anthropic's prompt engineering guide states this plainly, and it's the best single piece of advice in the whole field:

> **Show your prompt to a colleague with minimal context on the task and ask them to follow it. If they'd be confused, Claude will be too.**

Language models are powerful, but they don't have the context you have. They don't know what domain you're in, what your customers sound like, what "the usual format" means, or what the user is actually trying to accomplish. Every prompt should be readable as a standalone request by someone seeing the task for the first time. If your prompt assumes shared context that the model doesn't have, it will underperform.

A corollary: **explain *why* when you give instructions.** "Don't use ellipses" is weaker than "Don't use ellipses — your response will be read by a text-to-speech engine that doesn't know how to pronounce them." The model is smart enough to generalize from the reason, and will handle edge cases you didn't explicitly think of.

---

## Why prompt engineering is mostly a numbers game

You might assume that the best prompts come from careful intuition — wordsmiths who know exactly how to phrase things. In practice, the best prompts come from **evaluation loops**. You write a prompt, you run it against 20-100 test cases, you see which ones fail, you modify the prompt, you run it again. Repeat.

This is the single most important habit you'll build in this module:

1. **Never claim a prompt works without an eval set.** "It worked when I tried it" is not a prompt working. It's your memory of one run.
2. **Eval your prompts on real data, not vibes.** Pull actual user inputs from your application (or representative synthetic ones) and measure how often the model produces the right answer.
3. **Iterate by changing one thing at a time.** If you change the system prompt *and* add examples *and* switch to chain-of-thought all at once, you don't know which change helped.
4. **Keep old prompts.** Prompts are code. Version them. When something regresses, you want to know when and why.

Lesson 12 goes deep on writing prompt evals. For now, just internalize that every technique in this module — few-shot, chain-of-thought, role prompting, XML tags — is useful only insofar as it measurably improves your eval scores on *your* task. The techniques are tools. The eval is the test of whether the tool helped.

---

## A preview of what you'll learn

Here's where each technique in this module earns its place, so you can recognize them when they come up:

- **Structure (Lessons 2, 3, 8):** when you need the model to produce output in a reliable shape
- **Few-shot (Lesson 4):** when you can show the model examples of the exact pattern you want
- **Chain-of-thought (Lesson 5):** when the task has multiple reasoning steps and the model would skip them otherwise
- **Advanced reasoning (Lesson 6):** when CoT isn't enough — complex planning, branching exploration, tool use
- **Role prompting (Lesson 7):** when you want a consistent persona, tone, or expertise framing
- **Prompt caching patterns (Lesson 9):** when you need to reuse the same context across many calls cheaply
- **Multimodal (Lesson 10):** when images (or other media) are part of the input
- **Guardrails against injection (Lesson 11):** when you're putting LLMs in production with untrusted input
- **Evaluation (Lesson 12):** always. The habit that makes the rest work.
- **Frameworks (Lesson 13):** when you're ready to systematize prompting beyond hand-written strings

---

## Common pitfalls (even at this introductory level)

- **Being vague about the output.** "Summarize this" invites whatever the model feels like producing. "Summarize this in 3 bullet points, each ≤15 words, as JSON with a `summary` key" gives you something you can parse and measure.
- **Over-prompting.** The instinct once you learn techniques is to pile them all on. `You are an expert in X. Think step by step. Use examples. Return in JSON. Be concise. Be thorough. Consider multiple perspectives.` Contradictory instructions confuse the model and usually degrade quality. Simpler is often better — add complexity only when you can prove it helps.
- **Treating prompts as static.** A prompt that works great today may fail next month when the provider updates their model. Re-run your eval every time you upgrade.
- **Assuming prompts transfer across providers.** A prompt optimized for Claude may behave differently on GPT or Gemini. Always re-evaluate when porting.
- **Confusing "the model is wrong" with "my prompt is bad."** When the output is wrong, your first hypothesis should be "my prompt underspecified the task," not "the model can't do this." Only after you've exhausted prompt variations should you switch models or conclude the task is infeasible.
- **Assuming if one prompt is good, a longer one is better.** Length is not quality. Every extra word either adds clarity or adds noise. Longer prompts cost more (Lesson 11 of Module 1) and can distract the model from what actually matters.

---

## What to remember from this lesson

- Prompt engineering is how you activate the right region of a pre-trained model's learned distribution — it's the highest-leverage skill for practical LLM work.
- Small phrasing changes produce large quality changes. Don't trust intuition; test.
- There's a ladder from "just ask" to "DSPy pipelines." Most value is at Level 2 (structured prompts) and Level 3 (few-shot + CoT).
- Anthropic's golden rule: if a colleague with no context would be confused by your prompt, the model will be too.
- Explain *why* you're giving an instruction. The model generalizes from the reason.
- Every prompt you care about needs an eval set. "It worked when I tried it" is not a measurement.
- Techniques are tools. Evals are the test of whether the tool helped.

Next chapter: the mechanics of the message format every API uses — system, user, assistant — and why `system` is the most important field you've probably been ignoring.

---

## References

- Anthropic, *Prompt engineering overview*. https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview
- Anthropic, *Prompting best practices for Claude*. https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices
- OpenAI, *Prompt engineering guide*. https://platform.openai.com/docs/guides/prompt-engineering
- DAIR.AI, *Prompt Engineering Guide*. https://www.promptingguide.ai/
- Schulhoff et al. (2024), *The Prompt Report: A Systematic Survey of Prompting Techniques*. https://arxiv.org/abs/2406.06608
- DeepLearning.AI, *ChatGPT Prompt Engineering for Developers* (Andrew Ng and Isa Fulford). https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/

---

[← Back to Prompt Engineering](../README.md) | [Next → Lesson 2: Message Structure](02-message-structure.md)
