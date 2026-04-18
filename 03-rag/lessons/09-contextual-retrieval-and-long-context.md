# 09 — Contextual Retrieval and Long-Context Tradeoffs

> Contextual retrieval prepends an LLM-generated context blurb to each chunk before embedding; with prompt caching it costs about a dollar per million document tokens and cuts retrieval failures by a third, and for corpora that fit in the window, long-context prompting with caching sometimes beats chunked retrieval entirely.

Of every technique in this module, contextual retrieval is the one that changed the most between 2023 and 2026. It was published by Anthropic in September 2024, it works better than any other drop-in chunking improvement, and because prompt caching made the cost trivial it has become the default strategy for serious RAG systems. This lesson covers how it works, when to use it, and the parallel question — when your corpus is small enough that you do not actually need RAG at all and long-context prompting with caching is simply better.

## The problem contextual retrieval solves

Go back to the failure mode from lesson 04: chunks lose their context when you isolate them. A standalone chunk says "The company's revenue grew by 3% over the previous quarter." Which company? Which quarter? A human reader who knows the whole document can answer immediately; the chunk by itself cannot be indexed reliably, because there is no token in it that matches a query like "Acme Q3 2025 revenue."

The embedding model also suffers. It projects the chunk into a position in vector space based only on the tokens present. If the chunk contains no company name, the vector does not encode which company the chunk is about, and a query naming the company will not find it.

This is the single biggest structural limitation of chunking: **the unit you index loses the context that would help it be retrieved correctly.** You can partially fix it with larger chunks (you lose precision), small-to-big retrieval (helps but doesn't address indexing), or metadata (helps but requires schema). Contextual retrieval solves it at the source.

## How contextual retrieval works

For every chunk at index time, ask a fast LLM to read the **entire source document plus the chunk** and write a short contextual blurb that situates the chunk within the document. Then prepend the blurb to the chunk before embedding and indexing.

The prompt Anthropic used:

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within
the overall document for the purposes of improving search retrieval
of the chunk. Answer only with the succinct context and nothing else.
```

The generated blurb is 50–100 tokens. A chunk that was:

> The company's revenue grew by 3% over the previous quarter.

becomes, after prepending:

> This chunk is from Acme Corp's Q3 2025 10-Q filing, in the Management Discussion section, describing quarter-over-quarter revenue change from Q2 to Q3 2025.
>
> The company's revenue grew by 3% over the previous quarter.

Now the embedding captures "Acme," "Q3 2025," "revenue," and "quarter-over-quarter" even though those tokens were not in the original chunk. A BM25 index over the contextualised text also matches queries using any of those words. Both dense and sparse retrieval benefit at no runtime cost — the context is generated once, at index time.

## The numbers

Anthropic's public experiments (multiple domains: codebases, fiction, arXiv papers, scientific publications; top-20 retrieval with Gemini Text-004 embeddings):

| Configuration | Failure rate | Relative improvement |
|---|---|---|
| Baseline: dense embeddings + BM25 | 5.7% | — |
| Contextual embeddings only | 3.7% | −35% |
| Contextual embeddings + contextual BM25 | 2.9% | −49% |
| Contextual embeddings + contextual BM25 + reranker | 1.9% | −67% |

Two-thirds fewer retrieval failures from one chunking change plus one reranker. That is an enormous quality move. If your current RAG hit rate is 85%, contextual retrieval realistically takes you to 90–93% on the same corpus and index.

## The cost, and why prompt caching matters

If you naively generate context for every chunk by sending the full document in the prompt every time, the cost is ruinous. For a 8000-token document with 10 chunks, each chunk request would cost 8000 input tokens + 100 output tokens. Ten chunks = 80,000 input tokens per document. At Haiku-scale pricing (~$0.25/M input), that's $0.02 per document — meaning a million-document corpus would cost $20,000 to index.

**Prompt caching makes this trivially cheap.** You send the full document once with `cache_control: ephemeral` set on the document portion, and every subsequent chunk request for the same document reads the cached document at roughly 10% of the normal input cost. The effective cost drops by about 90%.

Using Anthropic's numbers: with 800-token chunks, 8,000-token documents, 50-token instructions, 100 tokens of output, the amortised cost is **about $1.02 per million document tokens**. For a typical corpus (say, 50 million tokens across 6,000 documents), indexing with contextual retrieval costs about $50. You will spend ten times that on the first week of LLM generation serving real users.

The code structure:

```python
def generate_context(document_text: str, chunk_text: str) -> str:
    response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>\n{document_text}\n</document>",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": CONTEXT_PROMPT.format(chunk_text=chunk_text),
                    },
                ],
            }
        ],
    )
    return response.content[0].text.strip()

def contextualise_document(document: Document) -> list[Chunk]:
    chunks = chunk_text(document.text, size=800, overlap=100)
    out = []
    for i, chunk_text in enumerate(chunks):
        context = generate_context(document.text, chunk_text)
        contextualised = f"{context}\n\n{chunk_text}"
        out.append(Chunk(
            document_id=document.id,
            chunk_index=i,
            chunk_text=chunk_text,
            contextualised_text=contextualised,
        ))
    return out
```

Store both the original `chunk_text` (for display and generation) and the `contextualised_text` (for embedding and BM25). Embed and BM25-index the contextualised version; when you retrieve and pass chunks to the LLM at query time, pass the original text so you are not wasting generator-side tokens on the index context.

## Implementation practicalities

- **Model choice.** Claude Haiku 4.5 is the default. Any fast, cheap model works; the task is simple and does not need frontier reasoning. Haiku is the right cost/quality point in 2026.
- **Document size limits.** The full document must fit in the model's window. Claude Haiku has a 200k window; most documents fit. For very long sources (books, codebases, massive PDFs), either split the document into logical sections first and contextualise within each section, or use a 1M-window model.
- **Domain-specific prompts.** Anthropic's generic prompt works well. If your corpus has specialised vocabulary, you can add a glossary or style guide to the prompt — the paper showed small additional gains from this.
- **Caching granularity.** The cache key is the document hash. Keep the document blob byte-identical between chunk requests so you hit the cache reliably.
- **Re-indexing.** If the source document changes, you need to regenerate context for every affected chunk. This is why you store a content hash per chunk and skip unchanged ones.

## When not to use contextual retrieval

Two situations:

1. **Your corpus is small enough to fit in the window.** If the whole knowledge base is under 200k tokens, just put it in the prompt, use prompt caching, and skip retrieval entirely. You pay cache-read rates on every turn (~10% of normal) and the model sees all of the evidence at once. We will come back to this in a moment — it is the "long context with caching" branch.
2. **Your chunks are already self-contained.** FAQs, sentence-per-chunk indexes, tables where each row is a complete fact. Contextual retrieval adds little when the chunk itself is already a complete answer.

For everything in between — moderate-to-large corpora, prose-heavy documents, multi-entity reports — contextual retrieval is the current default.

## The long-context tradeoff

The 2026 reality: context windows are getting absurd. Claude Opus 4.6 is 1M tokens. Gemini 3.1 Pro is 2M. GPT-5.4 is 1M. For corpora under about 200k tokens, you have a choice:

- **Traditional RAG:** chunk, embed, index, retrieve, generate. Complex pipeline, but scales.
- **Long-context with prompt caching:** put the whole corpus in the prompt, cache it, ask questions directly. No chunking, no embedding, no vector DB. Dead simple.

Long context wins when:

- Your corpus is static or changes rarely (caches stay warm).
- The whole corpus comfortably fits in the window with headroom.
- You want the model to see everything and reason across the entire corpus, not just the top-k chunks (multi-hop questions become free).
- You value simplicity — one prompt, one API call, no infra.

RAG wins when:

- Your corpus is too big.
- Latency matters and you cannot afford to process full-corpus tokens per turn.
- You need to serve many users with isolated per-user data.
- You need provenance at a fine granularity.
- Costs are dominated by tokens, not by the complexity of the system.

**Practical threshold:** under 50k tokens, prefer long context. 50k–200k tokens, either; run the numbers. Above 200k, RAG is still the only option. Keep in mind that the context cost scales linearly with context size even after caching, so "just put everything in the prompt" can become expensive at scale — measure cost per 1000 queries for each approach before committing.

## Lost in the middle — still relevant

Even with a 2M-token window, the "Lost in the Middle" phenomenon from Liu et al. 2023 still applies. Models pay attention to the start and end of long contexts more than the middle. The practical fix for RAG is:

- After reranking, order retrieved chunks so the most relevant ones are at the start and end of the context block, least relevant in the middle. This "sandwich" ordering outperforms naive ranked ordering for the same chunks.
- For long-context prompts, place the most important content (the user query, the system instructions, the most critical document sections) at the beginning or the end.

Modern models are better at this than the 2023 generation, but the effect has not disappeared. A simple reordering costs nothing and measurably improves answer quality on harder questions.

## A hybrid of RAG and long context

The emerging pattern for 2026: **use RAG to find the most relevant documents, then pass full documents — not chunks — to a long-context model.** You get the precision of retrieval with the coherence of full-document reading.

```
Query → RAG pipeline → top 3 documents (not chunks) → long-context LLM → answer
```

This works when each document is under ~100k tokens (so 3 of them fit) and you care more about coherence within each document than about fine-grained passage retrieval. It is the right architecture for things like "find the right contract and answer a question about it" where the chunked answer often loses cross-clause context.

## What to remember

- Contextual retrieval prepends an LLM-generated context blurb to each chunk at index time. Embed and BM25-index the contextualised version; keep original text for generation.
- Two-thirds fewer retrieval failures when paired with BM25 and a reranker. The biggest drop-in improvement you can make.
- Prompt caching makes the index cost trivial — about $1 per million document tokens.
- For corpora under ~200k tokens, prefer long-context prompting with caching. It is simpler and often better.
- "Lost in the middle" still applies. Order retrieved chunks most-relevant-first-and-last.
- A hybrid pattern — retrieve documents, not chunks, then long-context read — is gaining traction for coherent-document workloads.

## References

- Anthropic, *Introducing Contextual Retrieval*. https://www.anthropic.com/news/contextual-retrieval
- Anthropic, *Prompt caching*. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Liu et al. 2023, *Lost in the Middle: How Language Models Use Long Contexts*. https://arxiv.org/abs/2307.03172
- Anthropic Cookbook, *Contextual Retrieval example*. https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings
- Google, *Gemini long context engineering*. https://ai.google.dev/gemini-api/docs/long-context
- OpenAI, *Prompt caching guide*. https://platform.openai.com/docs/guides/prompt-caching
