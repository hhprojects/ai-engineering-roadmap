# 2 — Prompt Engineering

**Goal:** Build the instincts, techniques, and tooling to get reliable, cost-effective, and safe output from any LLM — from casual chat to production agent pipelines.

This module is a self-contained textbook. Like Module 1, every chapter synthesizes material from canonical sources and ends with a references section so you can dig deeper. You don't need to click any external links to complete the module, but the links are there if you want to.

Budget: **~15-20 hours** of reading + ~20-25 hours for the three projects.

---

## How to use this module

1. **Read the lessons in order.** They build on each other — Lesson 6 (advanced reasoning) assumes you know chain-of-thought from Lesson 5, and Lesson 13 (frameworks) assumes you know why you'd want structure from Lessons 8 and 12.
2. **Do Project 1 alongside Lessons 1-3.** It's the easiest and gives you the CLI you'll use to test techniques from later lessons.
3. **Do Project 2 alongside Lesson 8.** Structured outputs make the most sense when you've seen the problem they solve.
4. **Do Project 3 after Lesson 12.** It requires evaluation — do not attempt it without evals.
5. **Keep a notes file.** Every time a technique surprises you or a prompt fails unexpectedly, write it down. That's the material you'll retain.

---

## 📖 Lessons

| # | Lesson | Focus | Est. time |
|---|---|---|---|
| 1 | [What Is Prompt Engineering?](lessons/01-what-is-prompt-engineering.md) | Why prompts matter, the golden rule, the ladder of techniques | 25 min |
| 2 | [Message Structure and System Prompts](lessons/02-message-structure.md) | Roles, the instruction hierarchy, caching implications | 30 min |
| 3 | [Anatomy of a Good Prompt](lessons/03-anatomy-of-a-good-prompt.md) | Task, context, examples, format, tone — in order | 25 min |
| 4 | [Zero-Shot, Few-Shot, In-Context Learning](lessons/04-zero-shot-few-shot.md) | When to use examples, how many, which ones | 30 min |
| 5 | [Chain-of-Thought Prompting](lessons/05-chain-of-thought.md) | Step-by-step reasoning, self-consistency, CoT on reasoning models | 30 min |
| 6 | [Advanced Reasoning Techniques](lessons/06-advanced-reasoning.md) | Tree-of-thoughts, least-to-most, ReAct, reflexion, chaining | 35 min |
| 7 | [Role Prompting and Tone Control](lessons/07-role-prompting.md) | Personas, audience framing, tone by example | 20 min |
| 8 | [Structured Outputs in Practice](lessons/08-structured-outputs.md) | Schema design, Pydantic, Instructor, XML tags, retries | 35 min |
| 9 | [Templates and Caching Patterns](lessons/09-templates-and-caching.md) | Versioning prompts, Jinja2, prompt caching, compaction | 30 min |
| 10 | [Multimodal Prompting](lessons/10-multimodal-prompting.md) | Images in prompts, OCR-first pipelines, crop tools | 25 min |
| 11 | [Prompt Injection and Guardrails](lessons/11-prompt-injection.md) | Attacks, why defenses fail, defense in depth, dual-LLM | 35 min |
| 12 | [Evaluating and Iterating on Prompts](lessons/12-evaluating-prompts.md) | Eval sets, metrics, LLM-as-judge, regression testing | 35 min |
| 13 | [Prompt Programming Frameworks](lessons/13-prompt-programming-frameworks.md) | DSPy, Instructor, BAML, LangChain, LlamaIndex survey | 30 min |

---

## 🔨 Projects

| # | Project | Difficulty | Time | Cost | Prereq lessons |
|---|---|---|---|---|---|
| 1 | [Prompt Playground](projects/01-prompt-playground.md) | 🟢 Beginner | 4-6 hrs | ~$2 | 1-3, 7, 12 |
| 2 | [Structured Data Extractor](projects/02-structured-extractor.md) | 🟡 Intermediate | 6-8 hrs | ~$2-3 | 3-4, 8, 12 |
| 3 | [Multi-Model Router](projects/03-multi-model-router.md) | 🟠 Advanced | 10-14 hrs | ~$4-6 | 7-8, 11-12 + M1 L13 |

All three projects use the same skeleton: prerequisites, setup, runnable starter scaffolding, requirements, stretch goals, evaluation rubric, common pitfalls, cost estimate, and deliverables.

---

## Supplementary resources

If you want to go deeper than the lessons, these are the canonical sources. Most of the lesson content is synthesized from them.

### Core prompting guides
- [Anthropic — *Prompting best practices for Claude*](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices) — the most thorough provider guide
- [OpenAI — *Prompt engineering guide*](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI's official reference
- [DAIR.AI — *Prompt Engineering Guide*](https://www.promptingguide.ai/) — comprehensive open-source catalog
- [Schulhoff et al. — *The Prompt Report*](https://arxiv.org/abs/2406.06608) — the systematic survey, 58 techniques catalogued

### Structured outputs and frameworks
- [Instructor — *Structured outputs for Python*](https://python.useinstructor.com/)
- [DSPy — *Programming language models*](https://dspy.ai/)
- [BAML — *Boundary Markup Language*](https://docs.boundaryml.com/)
- [LangChain](https://python.langchain.com/docs/introduction/) / [LlamaIndex](https://docs.llamaindex.ai/)

### Evaluation
- [Hamel Husain — *Your AI product needs evals*](https://hamel.dev/blog/posts/evals/) — the essay every engineer should read
- [Eugene Yan — *LLM patterns and evals*](https://eugeneyan.com/writing/llm-patterns/)
- [OpenAI Evals](https://github.com/openai/evals) / [Braintrust](https://braintrust.dev/) / [Langfuse](https://langfuse.com/) / [promptfoo](https://promptfoo.dev/)

### Security
- [Simon Willison — *Prompt injection tag*](https://simonwillison.net/tags/prompt-injection/) — the reference for practitioners
- [OWASP — *Top 10 for LLM Applications*](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Greshake et al. — *Indirect Prompt Injection*](https://arxiv.org/abs/2302.12173)

### Videos and courses
- [DeepLearning.AI — *ChatGPT Prompt Engineering for Developers*](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) — Andrew Ng + Isa Fulford, free
- [DeepLearning.AI — *LangChain for LLM Application Development*](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- [Lilian Weng — *Prompt Engineering*](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) — blog-length survey

---

## Key concepts — the exit checklist

By the time you finish this module, you should be able to explain all of these in plain language:

- [ ] Why prompts matter — what the model is actually doing when it "follows" instructions
- [ ] The difference between system, user, and assistant roles, and the instruction hierarchy
- [ ] The five ingredients of a good prompt: task, context, examples, format, tone
- [ ] When to use zero-shot, when to use few-shot, and how many examples to pick
- [ ] How in-context learning works and why the specific examples matter less than the format
- [ ] What chain-of-thought prompting is and when it helps vs. hurts
- [ ] The difference between CoT and reasoning models; when to use which
- [ ] Self-consistency, tree-of-thoughts, least-to-most, ReAct, reflexion, and which one to reach for in which scenario
- [ ] How role prompting shapes tone and judgment, and why specifics beat grand titles
- [ ] How to design Pydantic schemas that extract well: enums, escape hatches, validators
- [ ] The difference between JSON schema, XML tags, and when to use each
- [ ] How prompt caching works on each major provider and how to structure prompts to cache
- [ ] How to prompt vision models and the tricks that improve accuracy (image-first, crop tool, quote before answer)
- [ ] What prompt injection is, why it's fundamentally unsolved, and the defense-in-depth patterns you need
- [ ] How to build an eval set, which metrics to use, and how to run LLM-as-judge well
- [ ] What DSPy/Instructor/BAML give you and when each is worth adopting

If any of these feel shaky, return to the corresponding lesson.

---

## A note on staleness

The LLM landscape moves fast. The lessons name specific model versions (Claude Opus 4.6, GPT-5.4, Gemini 3.1, etc.) as of **early 2026**. By the time you read this, those names will have shifted. The *concepts* are stable; the *specific capabilities and prices* turn over every 3-6 months.

Treat model names as examples, not permanent facts. Check the current provider pages for up-to-date capabilities and pricing. The references section above links to always-current sources — bookmark them.

---

[← LLM Fundamentals](../01-llm-fundamentals/) | [Home](../README.md) | [Next → RAG](../03-rag/)
