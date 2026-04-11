# Paper Reading Agent

> A tool-using agent that turns arXiv papers into flashcards, quizzes, and summaries — with a built-in LLM-as-judge eval harness and an MCP server wrapper.

**Status:** 📋 Planning
**Target ship date:** TBD (allow ~1–1.5 weekends of focused work)
**Target repo:** `github.com/hhprojects/paper-reading-agent` (standalone, not inside this roadmap)

---

## Why This Project Exists

This is the resume-centerpiece build. It replaces MyFitLife on the resume and serves as the public, verifiable counterpart to the HTX MiniAgent + Learning Cards / Mock Quizzes work — which can't be linked publicly.

The project is deliberately scoped to hit the **highest-leverage Applied AI keywords in one shot**:

- Agents + tool use + planning loop
- Model Context Protocol (MCP)
- Retrieval-Augmented Generation (RAG) with pgvector
- LLM-as-judge evaluation rigor
- Production deployment (Docker + FastAPI)

---

## What It Does

1. Accepts an arXiv URL or uploaded PDF
2. **Plans** extraction steps using an agentic loop (Claude/GPT with tool use)
3. **Executes** tools: `pdf_extract`, `chunk`, `embed`, `arxiv_fetch`, `web_search`, `citation_lookup`
4. **Generates** a structured output: summary, N flashcards, M quiz questions with answers
5. **Self-evaluates** using an LLM-as-judge harness against a golden set
6. **Serves** via FastAPI + is exposed as an MCP server so Claude Desktop can call it as a tool

---

## Target Resume Bullet (preview)

> *Built an autonomous research-paper agent that plans, extracts, and generates study material (flashcards + quizzes) from arXiv papers using a Claude/OpenAI tool-use loop, pgvector semantic chunking, and an LLM-as-judge eval harness over an [N]-paper golden set — achieving ~[X]% flashcard accuracy and ~[Y]% coverage vs manual baselines. Exposed as a Model Context Protocol (MCP) server for Claude Desktop integration; deployed via FastAPI + Docker.*

Exact metrics get filled in once the eval harness runs. See [`RESUME_BULLET.md`](RESUME_BULLET.md) for the full template and fallback phrasings.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Everything in AI/ML lives here |
| Agent loop | Hand-rolled Claude/OpenAI tool-use loop | Demonstrates fundamentals, avoids framework lock-in — better interview signal than "I used LangChain" |
| LLM | Claude Sonnet (primary), GPT-4o-mini (fallback) | Both are on your Skills line |
| Embeddings | OpenAI `text-embedding-3-small` OR `BAAI/bge-small-en-v1.5` via HF | Start with OpenAI, migrate to local if cost matters |
| Vector store | PostgreSQL + pgvector | You already claim pgvector; avoids introducing a new DB |
| PDF extraction | `pymupdf` (fast) or `unstructured` (robust) | Start with pymupdf |
| API | FastAPI | Already on your resume |
| MCP server | Python MCP SDK | Thin wrapper around the agent |
| Deployment | Docker Compose → Railway or Render free tier | Cheap, fast, ATS-visible |
| Eval | Custom LLM-as-judge harness + small golden set | Matches your HTX eval work |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the component diagram and data flow.

---

## Project Files

- [`README.md`](README.md) — this file (vision + pitch)
- [`PLAN.md`](PLAN.md) — day-by-day milestone checklist
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design + tool schemas
- [`RESUME_BULLET.md`](RESUME_BULLET.md) — bullet drafts + metric placeholders

---

## Placement on Resume (once shipped)

Replaces **MyFitLife Fitness Application** under the Projects section. KakiAlert stays (BrainHack Finalist recognition is valuable). Final Projects section will read:

1. **Paper Reading Agent** ← this project (Applied AI centerpiece)
2. **KakiAlert Crisis Reporting Application** (full-stack + AI image rec + competition credit)

---

[← Capstone](../README.md) | [04 — Agents](../../04-agents/) | [Roadmap Home](../../README.md)
