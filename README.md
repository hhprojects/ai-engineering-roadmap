# AI Engineering Roadmap 2026

> A self-paced, project-driven course to go from Python developer to AI engineer — no GPU required.

You don't need a PhD or a $10k cloud bill to become an AI engineer. This roadmap takes you from software engineering fundamentals through production-grade AI systems, one hands-on project at a time.

Every section has curated readings, YouTube videos, and 2-3 projects at different difficulty levels. Pick the ones that match where you are. Build stuff. Ship it. That's how you learn.

---

## 🗺️ Roadmap Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Engineering Roadmap 2026                   │
├─────────────┬───────────────┬───────────────┬───────────────────┤
│ §0          │ §1            │ §2            │ §3                │
│ Foundations  │ LLM           │ Prompt        │ RAG               │
│ Python,     │ Fundamentals  │ Engineering   │ Retrieval-        │
│ APIs,       │ Transformers, │ Techniques,   │ Augmented         │
│ Docker      │ Tokens,       │ Structured    │ Generation,       │
│             │ Attention     │ Outputs       │ Vector Search     │
├─────────────┼───────────────┼───────────────┼───────────────────┤
│ §4          │ §5            │ §6            │ §7                │
│ Agents      │ Observability │ Production    │ Career &          │
│ Tool Use,   │ Tracing,      │ Deploy,       │ Portfolio         │
│ MCP,        │ Evals,        │ Security,     │ Blog, OSS,        │
│ Multi-Agent │ LLM-as-Judge  │ Scale         │ Demo Videos       │
├─────────────┴───────────────┴───────────────┴───────────────────┤
│ 🧵 Capstone: Build Your Own AI Assistant (optional thread)      │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Sections

| # | Section | What You'll Learn | Projects |
|---|---------|-------------------|----------|
| 0 | [Software Engineering Foundations](00-foundations/) | Python, FastAPI, Docker, async | 3 |
| 1 | [LLM Fundamentals](01-llm-fundamentals/) | Transformers, tokenization, model families | 3 |
| 2 | [Prompt Engineering & APIs](02-prompt-engineering/) | Prompt techniques, structured outputs, function calling | 3 |
| 3 | [RAG](03-rag/) | Vector search, chunking, hybrid retrieval, evaluation | 3 |
| 4 | [Advanced Agents](04-agents/) | Tool use, MCP, multi-agent orchestration | 3 |
| 5 | [Observability & Evaluation](05-observability/) | Tracing, eval suites, LLM-as-judge | 3 |
| 6 | [Production](06-production/) | Deployment, caching, security, monitoring | 3 |
| 7 | [Career & Portfolio](07-career/) | Portfolio polish, blogging, open source | — |
| 🧵 | [Capstone](capstone/) | Build Your Own AI Assistant | 1 evolving project |

---

## 🚀 How to Use This Roadmap

### Linear Path
Start at §0 and work through each section in order. The optional [capstone thread](capstone/) ties everything together — you'll build one project that grows with each section.

### Modular Path
Already know Python? Skip §0. Comfortable with transformers? Jump to §2. Each section is self-contained with its own readings and projects.

### Difficulty Levels

Every project is tagged with a difficulty level:

- 🟢 **Beginner** — Core concepts, guided structure. Start here if the topic is new to you.
- 🟡 **Intermediate** — More moving parts, less hand-holding. Good default for most learners.
- 🟠 **Advanced** — Production patterns, complex integrations. For when you want to go deep.

You don't need to do all three — pick the level that stretches you without breaking you.

---

## ✅ Prerequisites

- **Python basics** — variables, functions, classes, list comprehensions. If you can write a 100-line script, you're good.
- **Terminal comfort** — navigate directories, run commands, use git.
- **A text editor/IDE** — VS Code recommended, but use whatever you like.
- **No GPU required** — every project runs on a regular laptop using APIs or CPU-only training on tiny datasets.

---

## 💰 Cost Estimates

Most resources are free. API costs for all projects combined:

| Resource | Cost | Notes |
|----------|------|-------|
| OpenAI API | ~$5-15 total | For all projects combined |
| Anthropic API | ~$5-10 total | Can use free tier for some |
| Groq API | Free | Open-source model inference |
| Together.ai | Free tier | 25M tokens/month free |
| ChromaDB | Free | Local vector DB |
| Langfuse Cloud | Free tier | 50k observations/month |
| Cohere Reranker | Free tier | 1000 calls/month |
| Railway/Render | Free tier | For deployment projects |
| **Total estimate** | **~$10-25** | **For completing all projects** |

> 💡 You can minimize costs by using Groq (free) and Together.ai (free tier) for most experimentation, and only using paid APIs when projects specifically require them.

---

## 📄 License

This roadmap is open source. Use it, fork it, share it, adapt it for your team.

---

*Built with ☕ and curiosity. Last updated: March 2026.*
