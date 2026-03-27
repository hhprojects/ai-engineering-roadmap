# 2 — Prompt Engineering & APIs

**Goal:** Master prompt techniques, structured outputs, function calling, and multi-provider API patterns.

## Learning Objectives

- Apply systematic prompt engineering techniques (chain-of-thought, few-shot, system prompts)
- Extract structured data from LLMs using function calling and Pydantic
- Build applications that work across multiple LLM providers

---

## 📚 Readings & Resources

| Resource | Type | Why |
|----------|------|-----|
| [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) | Docs | Official, always up-to-date |
| [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) | Docs | Anthropic's official guide — excellent on system prompts and chain-of-thought |
| [Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/) | Guide | Comprehensive open-source reference with techniques catalog |
| [OpenAI Function Calling docs](https://platform.openai.com/docs/guides/function-calling) | Docs | Official structured output / tool calling reference |
| [Instructor library](https://github.com/jxnl/instructor) | GitHub | The best lib for structured outputs with Pydantic — study the examples |
| [DeepLearning.AI: ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) | 🎬 Course | Free, practical, Andrew Ng + Isa Fulford |

---

## 🔨 Projects

| # | Project | Difficulty | Description |
|---|---------|------------|-------------|
| 1 | [Prompt Playground](projects/01-prompt-playground.md) | 🟢 Beginner | Your own prompt testing tool across providers |
| 2 | [Structured Data Extractor](projects/02-structured-extractor.md) | 🟡 Intermediate | Extract JSON from messy text with function calling |
| 3 | [Multi-Model Router](projects/03-multi-model-router.md) | 🟠 Advanced | Smart routing: cheap models for easy tasks, premium for hard ones |

---

## Key Concepts

After completing this section, you should understand:

- The difference between system, user, and assistant messages
- Chain-of-thought prompting and when it helps
- Few-shot vs. zero-shot prompting strategies
- How function calling / tool use works under the hood
- Structured outputs with JSON mode and Pydantic
- Temperature, top-p, and how they affect generation
- Provider-specific quirks (OpenAI vs. Anthropic API differences)
- When to use prompting vs. fine-tuning

---

[← LLM Fundamentals](../01-llm-fundamentals/) | [Home](../README.md) | [Next → RAG](../03-rag/)
