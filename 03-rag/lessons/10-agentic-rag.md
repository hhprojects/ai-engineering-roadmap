# 10 — Multi-Hop and Agentic RAG

> When one retrieval round is not enough, let the model plan and execute multiple retrievals — either statically (decomposition) or dynamically (agentic loops with a search tool).

Single-shot RAG is a good default for "find the document that answers this question." It is a terrible default for questions whose answer lives in multiple documents, or where the relevant query is not obvious until you have already read something. For those, you need multi-hop retrieval — executing more than one retrieval round, with later queries informed by earlier results. In 2026 the dominant pattern for multi-hop is agentic: give the model a search tool and let it decide when to call it. This lesson covers both the planned (decomposition) and the agentic versions, and tells you when each is the right choice.

## When single-shot breaks

Classes of queries that a single retrieval cannot handle well:

- **Compound questions.** "What's the difference between the Pro and Team plans, and how do I upgrade?" needs chunks about both plans *and* chunks about upgrading.
- **Bridge questions (multi-hop).** "Who is the author of the library that Anthropic uses for prompt caching?" requires knowing (a) the library name, then (b) looking up its author.
- **Constraint-satisfaction.** "Find all the incidents from Q3 that were caused by database deadlocks." The model needs to filter, count, or enumerate across many chunks.
- **Comparison queries.** "Compare the pricing of all three plans over the last year." Needs per-plan chunks *plus* temporal filtering.
- **Open-ended research.** "Summarise what this codebase does." Needs iterative exploration, not a single top-k.

On any of these, naive retrieve-and-generate will produce a plausible but incomplete answer that mentions one plan's pricing and not the other, or gets the library right and the author wrong. Users hate this because the answer *looks* right.

## Static decomposition

The simplest multi-hop technique, and the one you saw briefly in lesson 08: ask an LLM to decompose the query into sub-questions, retrieve for each, then synthesise.

```
User: "What's the difference between Pro and Team, and how do I upgrade?"
  │
  ▼
Decomposer LLM
  │
  ├─► "What are the features of Pro plan?"
  ├─► "What are the features of Team plan?"
  └─► "How do I upgrade my plan?"
  │
  ▼ (retrieve for each, in parallel)
  │
Concatenate retrieved chunks
  │
  ▼
Generator LLM → combined answer
```

Static decomposition is:

- **Fast** (retrievals run in parallel).
- **Predictable** (fixed number of steps).
- **Cheap** (one decomposer call + N retrievals + one generator call).
- **Limited** (cannot handle cases where the second question depends on the answer to the first).

Use it as the next step after single-shot RAG. It covers the first class of failures (compound questions) almost entirely.

## Agentic retrieval

For bridge questions and open-ended research, static decomposition is not enough because the next query depends on what you learned from the last one. That is where agentic retrieval comes in: give the LLM a **search tool** and let it decide, turn by turn, whether to search again, read what it has, or produce an answer.

The shape of an agentic RAG loop:

```
System:
  You are a research agent. You have one tool: search(query).
  Call it when you need information. Stop when you can answer.

User: Who wrote the library Anthropic uses for prompt caching?

Agent (turn 1): I need to find what library Anthropic uses for prompt caching.
  → tool call: search("Anthropic prompt caching library")

Tool result: [top-3 retrieved chunks about Anthropic's prompt caching feature]

Agent (turn 2): These mention that prompt caching is built into the Claude API.
  It's an internal feature, not an external library. But the question asks about
  the *library*; let me search for the client library.
  → tool call: search("Anthropic Python SDK author")

Tool result: [top-3 retrieved chunks about anthropic Python SDK]

Agent (turn 3): The Python SDK's author is listed as the Anthropic team. The
  library is maintained by Anthropic. Final answer: the `anthropic` Python SDK
  is maintained by Anthropic itself; there is no external author.
```

The model made two searches, reasoned about what it found, and returned an answer that explicitly addressed the ambiguity in the question. Static decomposition could not have done this — the second search depended on the first.

## Implementing an agentic RAG loop

Every modern LLM API supports tool use. Here is the pattern with the Anthropic SDK:

```python
from anthropic import Anthropic
client = Anthropic()

SEARCH_TOOL = {
    "name": "search",
    "description": (
        "Search the knowledge base for passages relevant to a query. "
        "Returns up to 5 passages with source IDs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and include key terms.",
            }
        },
        "required": ["query"],
    },
}

def agentic_rag(user_question: str, retriever, max_turns: int = 6) -> str:
    messages = [{"role": "user", "content": user_question}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=[SEARCH_TOOL],
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return "".join(
                block.text for block in response.content
                if block.type == "text"
            )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "search":
                    results = retriever.retrieve(block.input["query"], k=5)
                    passages = "\n\n".join(
                        f"[{r['chunk_id']}] {r['text']}"
                        for r in results
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": passages,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        break  # unexpected stop reason

    return "Could not complete research within turn limit."
```

A few pragmatic details:

- **Set a turn limit.** Agentic loops can runaway on ambiguous queries. 4–8 turns is usually enough; log when you hit the limit so you can spot pathological queries in production.
- **Return chunk IDs with the tool result.** The agent can then cite specific chunks in its final answer, which is essential for provenance.
- **Keep tool results concise.** Each tool call appends to the conversation; long results blow up the context. Use your reranker inside the tool so the agent only sees the top 3–5 reranked chunks.
- **Prompt the agent to explain its reasoning.** A system prompt that says "think about what you know and what you need before searching" reduces the number of wasted searches.

## Latency and cost

Agentic retrieval is slower and more expensive than single-shot. A realistic comparison:

| Approach | Turns | Latency | LLM tokens |
|---|---|---|---|
| Single-shot RAG | 1 | ~1 s | ~2k |
| Decomposition (static) | 1 | ~1.5 s | ~3k |
| Agentic (2 tool calls) | 3 | ~4 s | ~6k |
| Agentic (4 tool calls) | 5 | ~8 s | ~12k |

For interactive user-facing queries, keep agentic loops short and stream partial results. For background research tasks (generating a report, summarising a case file), longer loops are fine.

Cost scales with the agent's chattiness. A Sonnet-class model doing 4 tool calls costs ~$0.05 per query. If you are running 10,000 queries a day that is $500/day, which is real money. Consider a cheaper model (Haiku, GPT-5-mini) for the agent loop and a more expensive model only for the final answer synthesis if quality needs it.

## Self-RAG and reflection

Two patterns worth knowing about, though they are less common in production:

- **Self-RAG** (Asai et al. 2023) trains the model to decide when to retrieve and to evaluate whether retrieved passages are useful. In practice, modern frontier models already do this well enough with prompting; you rarely need the fine-tuned variant.
- **Reflection** asks the model to critique its own answer after generation and re-retrieve if the critique finds gaps. Simple to implement as one additional turn; helps on hard questions at the cost of doubling latency.

Both are variants on the same theme: let the model decide when it has enough information. In 2026 the main production pattern is vanilla agentic RAG with a search tool, because frontier models plan and self-evaluate well enough out of the box.

## When to use which

A decision tree:

- Is the question simple and factual? → **Single-shot RAG** with hybrid + reranking.
- Is the question compound but all sub-questions are independent? → **Static decomposition.**
- Is the question a bridge, multi-hop, or open-ended research query? → **Agentic RAG.**
- Is the query super hard and you cannot tell in advance? → **Start with single-shot, escalate to agentic** based on a simple classifier ("does this query need multi-step search?") — one cheap LLM call routes to the right pipeline.

## Query planning vs. agentic

One more useful distinction. "Query planning" usually means a single plan generated in advance (static decomposition). "Agentic" means the plan adapts turn by turn based on results. You can combine them: let the agent make a plan on the first turn, execute it in parallel, then iterate only if the results are insufficient. This is the pattern LlamaIndex and LangGraph default to.

## Common mistakes

- **Unbounded agent loops.** Always set a turn limit. Always log when you hit it.
- **No chunk IDs in tool results.** The agent cannot cite what it cannot name.
- **Raw retrieval output in tool results.** Rerank inside the tool and return only 3–5 high-quality chunks. Otherwise context fills with noise by turn 3.
- **Agentic where static decomposition would do.** Decomposition is cheaper, faster, and covers a huge fraction of real queries. Only escalate when needed.
- **Forgetting end-to-end evaluation.** Your eval from lesson 05 focused on retrieval hit rate. Agentic RAG needs end-to-end answer quality evaluation (lesson 12) because the agent can find the right chunks and still arrive at a wrong answer.

## What to remember

- Single-shot retrieval breaks on compound, bridge, multi-hop, comparison, and open-ended queries.
- Static decomposition handles compound questions. Agentic RAG handles the rest.
- Agentic RAG gives the model a `search` tool and lets it iterate. Every modern LLM API supports this.
- Set turn limits, return chunk IDs in tool results, rerank inside the tool.
- Agentic is more expensive and slower — route only the queries that need it there.
- Use cheaper models for the agent loop; reserve frontier models for final synthesis when quality matters.

## References

- Asai et al. 2023, *Self-RAG: Learning to retrieve, generate, and critique*. https://arxiv.org/abs/2310.11511
- Anthropic, *Tool use with Claude*. https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- LlamaIndex, *Agentic RAG patterns*. https://developers.llamaindex.ai/python/framework/use_cases/agents/
- LangGraph, *Plan and execute agents*. https://langchain-ai.github.io/langgraph/
- Trivedi et al. 2022, *Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions (IRCoT)*. https://arxiv.org/abs/2212.10509
- Shinn et al. 2023, *Reflexion — Language agents with verbal reinforcement learning*. https://arxiv.org/abs/2303.11366
