# 🧵 Capstone: Build Your Own AI Assistant

> An optional evolving project that ties the entire roadmap together. Each section adds a new layer to the same application.

## Overview

Instead of building separate projects for each section, the capstone thread has you build **one project that grows** as you progress through the roadmap. By the end, you'll have a fully functional, production-deployed AI assistant — and a portfolio centerpiece.

This is optional but highly recommended. Linear learners will find it especially satisfying.

---

## Progression

| Section | What You Add | Result |
|---------|-------------|--------|
| [0 — Foundations](../00-foundations/) | Python CLI skeleton with SQLite storage | Working CLI with conversation history |
| [1 — LLM Fundamentals](../01-llm-fundamentals/) | LLM API integration | CLI that talks to GPT/Claude |
| [2 — Prompt Engineering](../02-prompt-engineering/) | System prompts, structured outputs, streaming | Polished conversational experience |
| [3 — RAG](../03-rag/) | RAG over your personal notes/docs | Assistant that knows your stuff |
| [4 — Agents](../04-agents/) | Tools (web search, file management, calendar) | Agent that can take actions |
| [5 — Observability](../05-observability/) | Langfuse tracing + eval suite | Observable, measured quality |
| [6 — Production](../06-production/) | Deploy as web app with auth, caching, guardrails | Production-ready personal assistant |
| [7 — Career](../07-career/) | Write it up, open-source it | Portfolio centerpiece |

---

## How It Works

1. **Start in 0** — Build a simple CLI that stores conversations in SQLite. No AI yet, just the skeleton.
2. **Each section** — After completing a section's readings and at least one project, come back and add that section's concepts to your assistant.
3. **Incremental commits** — Each addition should be a meaningful commit. Track the evolution.
4. **Same repo** — Keep it all in one repository. The git history tells the story.

---

## Why This Matters

- **Compound learning** — Each layer reinforces the previous ones
- **Portfolio story** — "I built this from scratch, adding layers as I learned" is compelling
- **Real architecture decisions** — You'll face the same integration challenges production teams do
- **Interview gold** — Walk through the git history in an interview and explain every decision

---

## Tips

- **Don't over-engineer early** — 0 should be a simple CLI. Resist the urge to add AI on day one.
- **Commit often** — Each section addition should be 2-3 commits minimum (feature, tests, polish).
- **Write a CHANGELOG** — Document what you added and why at each stage.
- **Keep a decisions log** — "I chose ChromaDB over Pinecone because..." — this is interview material.

---

[← Career](../07-career/) | [Home](../README.md)
