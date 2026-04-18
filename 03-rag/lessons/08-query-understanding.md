# 08 — Query Understanding: Rewriting, Expansion, HyDE

> Users write queries for other humans, not for retrievers; query understanding transforms the raw query into one (or several) that your index actually knows how to answer.

A lot of RAG failures trace back to a mismatch between how users phrase questions and how information is written in your corpus. A user types "refund after trial," your chunks say "Cancellation during introductory period entitles customer to a full reimbursement," and the vector similarity is underwhelming. You can attack this by improving retrieval, but you can also attack it at the other end — by fixing the query itself before it hits the index. That is query understanding, and it is usually the second layer you add after reranking.

## The gap between queries and documents

Three structural mismatches cause most of the query-understanding work in production RAG:

1. **Vocabulary mismatch.** Users use plain words; documents use technical or domain-specific language. ("annual leave" vs. "paid time off," "laptop died" vs. "power delivery fault in the motherboard.")
2. **Intent ambiguity.** A short query can mean many things. ("reset" → password reset? factory reset? SQL `RESET`?)
3. **Compound questions.** One query covers two or three questions, each of which should hit a different chunk. ("How do I upgrade my plan and what's the difference between Pro and Team?")

Each failure mode has a different fix, and modern systems apply several of them simultaneously.

## Query rewriting

The simplest technique: before retrieval, ask a small LLM to rewrite the user's raw query into a form better suited for search. The rewrite can fix typos, expand abbreviations, resolve pronouns from conversation history, and replace colloquialisms with the terminology likely to appear in the corpus.

```python
REWRITE_PROMPT = """
You are a search query rewriter. Rewrite the user's question as a
concise, keyword-rich search query that is likely to match passages
in a technical knowledge base. Fix typos, expand abbreviations, and
resolve any pronouns using the conversation history.

Return ONLY the rewritten query, nothing else.

Conversation history:
{history}

User question: {query}
"""

def rewrite_query(query: str, history: list[dict]) -> str:
    messages = [{"role": "user", "content": REWRITE_PROMPT.format(
        history="\n".join(f"{m['role']}: {m['content']}" for m in history[-4:]),
        query=query,
    )}]
    response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=messages,
    )
    return response.content[0].text.strip()
```

Use a cheap, fast model for this (Haiku, GPT-5-mini, Gemini Flash). The latency adds 100–300 ms but you only pay tens of tokens per rewrite. The quality win on conversational RAG — where follow-up questions like "why?" or "what about the second option?" need history resolution — is large.

**Conversation-history resolution** is worth calling out specifically. A query like "What about Qdrant?" is meaningless without the prior turn. The rewriter should resolve it to "How does Qdrant compare to pgvector for hybrid search?" before retrieval. Skipping this step is the single most common cause of bad multi-turn RAG.

## Query expansion and multi-query retrieval

Instead of replacing the original query, expand it into several variants and retrieve for each. This gives you multiple "angles" on the same question and catches chunks that might have been written using different terminology.

```python
EXPAND_PROMPT = """
Generate 3 alternative phrasings of the following question that a
different user might use to ask about the same thing. Use different
vocabulary where possible. Output one per line.

Question: {query}
"""

def expand_query(query: str) -> list[str]:
    response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": EXPAND_PROMPT.format(query=query)}],
    )
    rewrites = [line.strip() for line in response.content[0].text.splitlines() if line.strip()]
    return [query] + rewrites  # always keep the original
```

For retrieval, you can either:

- **Pool candidates.** Run each expansion through hybrid search, then fuse results with RRF across all expansions (same trick as combining dense + sparse).
- **Parallel answer and vote.** Retrieve for each, generate an answer for each, and pick the majority. Heavier but sometimes effective for ambiguous queries.

Multi-query retrieval is a LangChain staple (`MultiQueryRetriever`) and shows up in most production stacks as the second query-understanding layer. The quality win is modest but reliable — expect 2–5 points of hit@5 on hard queries.

## Query decomposition

When users ask compound questions, a single retrieval cannot possibly work — the answer lives in multiple chunks about different topics, and the top-k gets "mixed" with low quality across both. The fix is to detect the compound structure and break the query into subqueries.

```python
DECOMPOSE_PROMPT = """
Break the user's question into independent sub-questions, one per line.
If the question is already simple, return it unchanged on a single line.

Question: {query}
"""
```

For a query like "How do I upgrade my plan and what's the difference between Pro and Team?" you get:

```
How do I upgrade my plan?
What is the difference between Pro and Team plans?
```

You then run both sub-queries through retrieval, concatenate the retrieved chunks, and let the LLM produce a combined answer. Query decomposition is the simplest form of multi-hop RAG (lesson 10). It is worth implementing as soon as your users start asking compound questions in production logs — you will spot them within a week of going live.

## HyDE: Hypothetical Document Embeddings

This is one of the more clever ideas in the RAG literature. The intuition: the user's query and the document that answers it are in *different* textual registers. "How do I reset my password?" is a question; the answer document says "To reset your password, navigate to Settings → Security → Reset password." Dense embeddings try to bridge this question/answer register gap, and they do it imperfectly.

HyDE (Hypothetical Document Embeddings, Gao et al. 2022) sidesteps the gap by asking an LLM to *invent* a plausible answer to the question, then embedding the fake answer and using it as the query vector. The fake answer may contain fabricated facts, but the embedding captures the *textual register of an answer*, which lives in the same neighbourhood as the real answers in your corpus.

```python
HYDE_PROMPT = """
Write a short, confident, plausible answer to the following question
as if it were extracted from a technical document. Two or three
sentences. Do not say "I'm not sure" or "it depends." Invent plausible
specifics if needed. We are using this text only to find matching
documents, not to actually answer the question.

Question: {query}
"""

def hyde_query_vector(query: str, embedder) -> list[float]:
    response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
    )
    fake_answer = response.content[0].text.strip()
    return embedder.embed(fake_answer)
```

You then use the HyDE vector for dense retrieval. You can keep the original query for BM25 (identifiers should not be fabricated) and fuse results with RRF.

**When HyDE helps:** short, conversational queries; corpora where the answers are much longer than the questions (technical documentation, legal text, research papers). Gao et al. showed significant improvements on TREC and MS MARCO without any supervised training.

**When HyDE hurts:** high-precision queries where the original wording matters; queries that contain identifiers or specific terms the model should not invent. Never HyDE a query like "show me the document with error code ERR-4472" — the fabricated answer might drift to a similar-looking code.

**When HyDE is redundant:** contextual retrieval (lesson 09) and good reranking (lesson 07) already address most of the question/answer register gap. On modern pipelines with contextual chunking and a reranker, HyDE adds about 1–3 points of hit@5 — nice to have, not essential.

## Query routing

When your system has multiple indexes (e.g., product docs, internal wiki, customer tickets), the *first* query-understanding decision is often "which index should this query go to?" This is query routing.

Two ways to do it:

1. **Metadata filters via auto-extraction.** Ask a small LLM to extract filters from the query: "Show me Q3 invoices from last year" → `{"doc_type": "invoice", "quarter": "Q3", "year": 2025}`. Then apply those as vector-DB filters.
2. **Retriever selection.** Ask a small LLM to classify which retriever to use based on the query: "code-search" vs. "prose-search" vs. "SQL-agent." Then dispatch to the right pipeline.

LlamaIndex calls this the `RouterQueryEngine`; LangChain has `MultiRetrievalQAChain`. The actual implementation is a few lines — an LLM call, a switch statement, then retrieval. The hard part is designing the routing rubric, which is a prompt-engineering exercise.

## Combining query understanding layers

You rarely need all of these. A typical production stack picks two or three:

- **Basic chatbot:** rewrite for conversation history only. One LLM call, huge win.
- **Intermediate:** rewrite + decomposition for compound questions. Two LLM calls for compound queries, one for simple.
- **Advanced:** rewrite + decomposition + HyDE + routing across multiple indexes. Four LLM calls, each on a cheap model. 300–600 ms added latency; significant quality gains on hard queries.

The cost is manageable because every step uses a cheap, fast model (Haiku, GPT-5-mini, Gemini Flash at ~$0.25 per million tokens). The whole query-understanding stack costs less than the reranker and far less than the generator.

## Evaluating query understanding

Use the same eval harness from lesson 05, but watch out for a trap: query understanding can improve your eval metrics in misleading ways if your eval set has queries that are already rewritten. If every eval query is clean, keyword-rich, and well-phrased, query rewriting buys you nothing — but your users' queries in production are none of those things.

**Build your eval set from real or realistic raw queries.** Typos, pronouns, fragments, compound questions. Otherwise you will optimise for a fantasy.

A before/after table for query rewriting should look like:

| Config | hit@5 on clean queries | hit@5 on raw queries |
|---|---|---|
| No rewrite | 0.89 | 0.68 |
| + Rewrite | 0.90 | 0.84 |
| + Decomposition | 0.90 | 0.87 |
| + HyDE | 0.91 | 0.88 |

The "clean queries" column barely moves; the "raw queries" column moves a lot. That is the right picture.

## Common mistakes

- **Rewriting every query.** Cost and latency cost apply to every turn. Rewrite only when the raw query is ambiguous or when conversation history exists; skip it for clearly-formed single-turn queries.
- **Over-decomposing simple queries.** The decomposer should return the original query unchanged for simple questions. Test this explicitly.
- **Running HyDE without fusion.** HyDE replaces the query vector, so if it drifts you lose exact-match behaviour. Run BM25 with the original query in parallel.
- **Rewriting away the identifiers.** "Find document ERR-4472" should NOT be rewritten to "Find document about error 4472." The rewriter needs a prompt that preserves exact identifiers; watch for this in eval.
- **Forgetting conversation history.** The single biggest UX gap in RAG chatbots. Solve it with a rewrite step that resolves pronouns and follow-ups.

## What to remember

- The user's raw query and the documents in your index live in different vocabularies. Query understanding bridges the gap.
- Rewrite for conversation history first — it is the cheapest win.
- Decompose compound questions into subqueries.
- HyDE generates a fake answer and embeds it; use on top of dense retrieval, keep BM25 on the original query.
- Route queries to the right index with metadata extraction or retriever classification.
- Build your eval set from raw, realistic queries. A clean eval set hides all the gains.

## References

- Gao et al. 2022, *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE). https://arxiv.org/abs/2212.10496
- LangChain, *MultiQueryRetriever*. https://python.langchain.com/docs/how_to/MultiQueryRetriever/
- LlamaIndex, *Query rewriting and decomposition*. https://developers.llamaindex.ai/python/framework/optimizing/production_rag/
- Microsoft, *Query rewriting for retrieval-augmented generation*. https://www.microsoft.com/en-us/research/blog/query-rewriting-for-rag/
- Ma et al. 2023, *Query rewriting for retrieval-augmented large language models*. https://arxiv.org/abs/2305.14283
- LlamaIndex, *Router query engine*. https://developers.llamaindex.ai/
