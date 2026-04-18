# 01 — What RAG Is and When to Use It

> RAG is the pattern of fetching relevant text at query time and stuffing it into the prompt so the model answers from that text instead of its parametric memory.

Retrieval-Augmented Generation has been the single most-deployed LLM pattern since 2023, and in 2026 it is still the default architecture for "chat with your docs," customer support copilots, internal search, and almost every vertical AI product that needs to answer questions about data the base model has never seen. Before you touch a vector database or an embedding API, you need to understand what problem RAG actually solves, what it does not, and when a simpler or different approach will serve you better.

## The problem RAG solves

A frontier model like Claude Opus 4.6 or GPT-5.4 has read an enormous amount of public text, but it has three structural limits that matter for real products:

1. **Knowledge cutoff.** The model stops learning at some training date. Anything after that — last week's product update, yesterday's incident report, the PR merged an hour ago — is invisible.
2. **Private data.** Your customer's invoice, your company's internal wiki, a user's personal notes. None of it was in the training corpus, and none of it should be.
3. **Provenance.** Even when the model "knows" something, it cannot reliably point to where it learned it. "Trust me" does not survive an audit, a legal review, or a user who wants to verify a claim.

You can attack all three problems by **providing the relevant text as part of the prompt itself**. The model then treats that text as the ground truth for the current turn and generates an answer grounded in it. That is RAG in one sentence: fetch the right snippets, put them in the prompt, ask the model to answer from them.

The trick is the fetching. You cannot stuff an entire 50-million-token wiki into every request — even with Claude's 1M token context window and Gemini 3.1 Pro's 2M window, that would be absurdly expensive and, as the "Lost in the Middle" paper showed, counterproductive. So RAG systems build an index once and at query time pull back only the 5 to 50 chunks most likely to contain the answer.

## The canonical architecture

Every RAG system, from a weekend prototype to a production platform like Perplexity or Notion AI, has the same four stages:

```
          INDEX-TIME                          QUERY-TIME
  ┌──────────────────────┐          ┌────────────────────────┐
  │ 1. Load source docs  │          │ 3. Retrieve top-k      │
  │ 2. Chunk + embed     │  ──────► │ 4. Generate answer     │
  │    store in index    │          │    with retrieved text │
  └──────────────────────┘          └────────────────────────┘
```

**Index time:** You walk the source corpus, split each document into chunks, compute an embedding for each chunk (a dense vector that captures semantic meaning — see lesson 02), and store the vectors plus the raw text in a vector database. This runs once and is re-run incrementally as sources change.

**Query time:** A user query arrives. You embed the query using the same embedding model, search the index for the nearest-neighbour chunks, and send those chunks plus the original question to an LLM with a grounding prompt that says "answer only from this context; if the context does not cover the question, say you do not know."

The naive version works surprisingly well on small, clean corpora. It also breaks in many entertaining ways on real data, and the rest of this module is about how to fix those breakages.

## RAG is not the only option

One of the most common mistakes junior engineers make in 2026 is to reach for RAG by reflex. Before you build anything, ask whether RAG is actually the right tool. There are four serious alternatives:

| Approach | When it wins | When it loses |
|---|---|---|
| **Long-context prompting** | Corpus fits in the model's window (under ~200k tokens in practice). One user, one document, conversational flow. You can cache the document with prompt caching and pay almost nothing per turn. | Millions of tokens. Multi-tenant knowledge bases. Data that changes constantly. |
| **Fine-tuning / continued pretraining** | You need the model to adopt a style, tone, format, or specialised vocabulary. You have thousands of high-quality examples. The knowledge is stable. | You need freshness, provenance, or per-user isolation. You do not have enough examples. |
| **Tool use / function calling** | The answer lives in a structured system (database, CRM, API) where you can query directly. You need live state, not semantic recall. | Your data is unstructured prose that nobody has a clean API for. |
| **RAG** | Large, unstructured corpus. Changes frequently. Need provenance. Answers must cite sources. | Source is trivially small (use long context) or the task is not really about retrieval (use fine-tuning or tools). |

A good instinct: **if you can point to the exact document that should answer the question, RAG is probably the right tool.** If the answer is implicit in a pattern across thousands of documents, a fine-tune or an analytical tool will often serve better. If the answer is one SQL query away, write the SQL.

A blended architecture is almost always the best production answer: tools for structured data, RAG for unstructured data, long-context caching for anything that fits in the window, and a frontier model routing between them.

## When RAG quietly fails

Even when RAG is the right choice, the naive implementation will disappoint you. Common failure modes worth naming up front, because almost every later lesson in this module exists to address one of them:

- **The relevant chunk is in the index but not in the top-k.** The embedding space buries it under noise. Fixes: better chunking, hybrid search (lesson 06), reranking (lesson 07), query rewriting (lesson 08).
- **The relevant chunk is retrieved but the model ignores it.** "Lost in the middle" — the model pays attention to the start and end of the context and skims everything between. Fixes: fewer, better chunks; reordering; contextual retrieval (lesson 09).
- **The chunk is too small to make sense.** "The company's revenue grew 3%" — which company, which quarter? Fixes: contextual chunking, small-to-big retrieval, parent-document retrieval (lessons 04 and 09).
- **The query asks about multiple things at once.** One retrieval round is not enough. Fixes: query decomposition, multi-hop RAG (lesson 10).
- **The answer looks right but is wrong.** The model hallucinates details the context does not support. Fixes: faithfulness evaluation, groundedness checks, Ragas (lessons 05 and 12).
- **Stale data.** The index was built last week; the source changed yesterday. Fixes: incremental indexing, freshness monitoring (lesson 13).

When you read a blog post that says "RAG is dead," what the author usually means is "naive RAG is not good enough." That is true. Modern production RAG is a stack of five or six techniques layered on top of the naive pipeline, and this module walks you through every layer.

## Jason Liu's levels as a roadmap

Jason Liu's [Levels of Complexity for RAG](https://jxnl.co/writing/2024/02/28/levels-of-complexity-rag-applications/) is the most useful mental model I know for thinking about where you are in a RAG project. He describes five levels:

1. **The basics** — file system, chunks, embeddings, simple top-k semantic search, basic LLM answering. You can build this in an afternoon.
2. **Structured processing** — async ingestion, better chunking, query rewriting, reranking, cited answers, streaming.
3. **Observability** — logging traces, tracking retrieval scores, segmenting metrics by user cohort, watching for cohort-specific failure.
4. **Evaluation** — a proper eval set, offline metrics for retrieval quality, LLM-as-judge for answer quality, a feedback loop.
5. **Understanding shortcomings** — clustering production queries to find capability gaps, rewriting the index or prompt based on what users actually ask.

The important insight: **you cannot skip levels**. A team that jumps straight to level 4 without observability spends weeks tuning metrics on a dataset that does not reflect real usage. A team that stops at level 1 ships a demo that looks great and breaks in production the first time a user asks something the team did not anticipate.

This module covers the techniques you need for levels 1 through 4. Level 5 is a habit, not a technique — you get there by shipping, watching, and iterating.

## What changed in 2026

A few things that did not exist (or were not practical) the last time you read a RAG tutorial:

- **1M+ token context windows** are now standard on frontier models. Anything under 200k tokens should seriously consider long-context-with-caching before reaching for a vector DB.
- **Prompt caching** cuts the cost of cached input tokens by roughly 90% and makes long-context RAG viable at scale. Anthropic, OpenAI, and Google all offer it.
- **Contextual retrieval** (Anthropic, late 2024) has become a default chunking strategy for any corpus large enough that long context does not fit.
- **Agentic retrieval** — letting the model call a search tool iteratively — has taken over from single-shot retrieval for harder questions. This is lesson 10.
- **Rerankers are cheap and fast.** Cohere's Rerank 4 and Voyage's rerankers run in tens of milliseconds and give you one of the highest-leverage quality wins in the stack.
- **Evaluation frameworks matured.** Ragas, TruLens, and LangSmith evals are no longer toys; you should have a real eval harness from day two of a RAG project.

## What to remember

- RAG fetches relevant text at query time and puts it in the prompt. That's it.
- The pipeline is load → chunk → embed → index (once); embed query → retrieve → generate (per turn).
- Ask whether RAG is the right tool before reaching for it. Long context, fine-tuning, and function calling each beat RAG in specific situations.
- Naive RAG is a starting point, not a destination. Every later lesson in this module is a fix for a specific failure of naive RAG.
- Work through Jason Liu's levels in order. Observability and evaluation come before clever tricks.
- In 2026, consider long-context + prompt caching for any corpus that fits in the window. Use RAG when it does not.

## References

- Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — the original RAG paper. https://arxiv.org/abs/2005.11401
- Anthropic 2024, *Introducing Contextual Retrieval* — the current state of the art for indexing. https://www.anthropic.com/news/contextual-retrieval
- Jason Liu, *Levels of Complexity: RAG Applications*. https://jxnl.co/writing/2024/02/28/levels-of-complexity-rag-applications/
- Anthropic, *Prompt caching* documentation — 90% cost reduction on cached input. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Pinecone, *What is Retrieval-Augmented Generation?* — clear conceptual intro. https://www.pinecone.io/learn/retrieval-augmented-generation/
- Liu et al. 2023, *Lost in the Middle: How Language Models Use Long Contexts*. https://arxiv.org/abs/2307.03172
