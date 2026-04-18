# 11 — Structured and Graph Retrieval (GraphRAG, SQL-RAG)

> When your data has structure — entities, relationships, rows, columns — vector search is the wrong tool; graph RAG and text-to-SQL give the model direct access to the structure and handle query types that semantic search cannot answer at all.

There is a class of questions that naive vector RAG cannot answer well no matter how much you tune it. "How many customers churned last quarter and what were the top reasons?" is not a retrieval problem, it is an aggregation problem. "Which papers cite Lewis et al. 2020 and also reference prompt caching?" is not a similarity problem, it is a graph traversal problem. For these, you need retrieval that respects the structure of your data. This lesson covers the two dominant structured-retrieval patterns in 2026: GraphRAG (for entity and relationship questions) and SQL-RAG / text-to-SQL (for structured data in relational databases).

## When structured retrieval wins

Vector RAG is good at "find me the passages that talk about X." It is bad at:

- **Counting and aggregation.** "How many support tickets mentioned shipping delays?"
- **Filtering by structured attributes.** "Show me contracts signed before 2025 with amounts over $100k."
- **Relationships across entities.** "Which employees reported to managers who reported to Carol?"
- **Global summaries.** "What are the main themes across this 10,000-document corpus?"
- **Temporal queries.** "What changed in the API between v3 and v4?"

The reason is that embeddings compress individual chunks into semantic space, losing the relational and numerical structure. A chunk that says "3 customers" and a chunk that says "500 customers" embed to similar vectors; the number is lost in the compression. No amount of reranking recovers it.

If the question is "what are customers saying about feature X," vector RAG is the right tool. If the question is "how many tickets about feature X were closed unresolved," you need structured retrieval.

## Two kinds of structure, two patterns

The structure might already be there (a database, a spreadsheet, a CRM) or it might be latent in unstructured text (mentions of people, places, and relationships in prose). Each case has a different pattern:

- **Structure is explicit → text-to-SQL / text-to-API.** Let the model generate a query in the structured language of your data store.
- **Structure is implicit → GraphRAG.** Extract an entity-and-relationship graph from the text at index time, then answer questions by traversing the graph.

You will see both in production systems, often side-by-side. Perplexity uses text-to-SQL for specific analytics-style queries and vector RAG for everything else. Microsoft GraphRAG exists for the implicit case.

## Text-to-SQL with an LLM

The pattern is straightforward and has been around since 2023: expose a read-only database connection, give the LLM the schema, and let it generate SQL that answers the question.

```python
SQL_TOOL_PROMPT = """
You are a SQL assistant. You have access to this schema:

{schema}

When the user asks a question, write a single SQL query that answers it.
Return ONLY the SQL query, no explanation. Use parameterised queries
where possible. If the question cannot be answered from this schema,
return the string "UNANSWERABLE".
"""

def run_sql_query(question: str, schema: str, db) -> str:
    sql = llm.generate(SQL_TOOL_PROMPT.format(schema=schema), question)
    if sql == "UNANSWERABLE":
        return "The data does not contain that information."
    rows = db.execute(sql)  # read-only connection; see safety notes
    return format_rows_as_markdown(rows)
```

Then you either return the rows directly or pass them back to a generator LLM to phrase the answer naturally.

**Making this safe in production:**

- **Read-only database role.** The account the LLM uses has SELECT privileges only. No INSERT/UPDATE/DELETE/DROP. This is non-negotiable; assume the LLM will be prompt-injected at some point and plan accordingly.
- **Query timeout and row limit.** `SET statement_timeout = 10s;` and add `LIMIT 1000` as a default. A misguided `JOIN` can take down your database.
- **Query validation.** Parse the generated SQL with a library like `sqlglot` and reject anything other than `SELECT`. This catches obvious escalation attempts before they reach the DB.
- **Schema card, not the whole DDL.** Pass a hand-curated description of tables and columns — one sentence per table, call out which columns are indexes or foreign keys. LLMs generate better SQL from a clear schema card than from raw DDL.
- **Few-shot examples.** A handful of (question, expected SQL) pairs in the prompt dramatically improve the LLM's output, especially for domain-specific terminology.
- **Retry on error.** When a query fails, pass the error back to the LLM for a single retry. Two failures in a row usually means the question is unanswerable.

The 2026 state-of-the-art for text-to-SQL is a Sonnet-class model with a good schema card and 3–5 few-shot examples. Accuracy on simple queries (single table, aggregation, simple joins) exceeds 90%; on complex multi-join queries with CTEs, it drops to 60–70%, which is why humans still write the hard queries.

## Query routing revisited

You almost never want the user's query to *always* hit the SQL path. Most queries are narrative; a few are structured. Route them with a classifier — one cheap LLM call that decides "this is a SQL question" vs. "this is a search question":

```
Classify the question:
  - SQL: needs aggregation, filtering, or lookups over structured data.
  - SEARCH: needs passages or prose from the knowledge base.
  - BOTH: needs both structured data and passages (e.g., "summarise the
    top 5 reasons customers churned last quarter").

Question: {query}
Respond with one word: SQL, SEARCH, or BOTH.
```

The router output dispatches to the right pipeline. For `BOTH`, you run both, then pass results to a final generator. This is exactly the LlamaIndex `RouterQueryEngine` pattern from lesson 08.

## GraphRAG: when the structure is implicit

Microsoft's GraphRAG (2024) tackles the harder case: the structure you need is implicit in unstructured text, and no database schema exists. GraphRAG builds the schema from the text automatically.

The indexing pipeline:

1. **Text units.** Split the corpus into chunks.
2. **Entity and relation extraction.** For each chunk, prompt an LLM to extract entities (people, places, concepts) and relations between them. The LLM returns a small "local graph" per chunk.
3. **Graph construction.** Merge the local graphs into a global knowledge graph. Nodes are entities, edges are relations, weights are co-occurrence counts.
4. **Community detection.** Apply hierarchical clustering (Leiden algorithm) to group densely connected entities into "communities" at multiple levels of granularity.
5. **Community summaries.** For each community, generate a short LLM summary of what that community is about — "this community contains 12 entities related to Anthropic's product roadmap."

At query time, GraphRAG has three search modes:

- **Local search** is "traditional" entity-centric retrieval: find the most relevant entities, pull their neighbourhood, use it as context.
- **Global search** uses the community summaries to answer broad questions like "what are the main themes in this corpus?" Each community answers the question partially; the model aggregates the partial answers.
- **DRIFT search** combines the two — start local, expand to community context as needed.

The killer application for global search is the kind of "corpus-level" question that vanilla RAG fails at: "summarise the key findings across all these 500 clinical trial reports." A top-k vector retriever samples only k of the 500; global GraphRAG reads community summaries covering all of them and produces an answer that reflects the whole corpus.

**Cost warning:** GraphRAG indexing is expensive. Every chunk goes through an entity-extraction LLM call, and community summarisation is another pass. Microsoft's published numbers put indexing cost at roughly 10–50× vanilla RAG. For static corpora this is a one-time bill; for dynamic corpora it can be painful.

**Quality warning:** LLM entity extraction is noisy. The graph will contain duplicates ("Anthropic" vs "Anthropic, Inc" vs "the Anthropic team"), missed relationships, and hallucinated facts. You need a dedup / canonicalisation step, and for production systems you need periodic human review.

## Hybrid: vector RAG + graph for neighbourhood expansion

A simpler and more popular 2026 pattern: run vector RAG as usual, but for each retrieved chunk, look up its entities in a graph and add the graph neighbourhood as additional context. This gives you some of GraphRAG's relational benefit without the full indexing cost.

```
1. Query → vector retrieval (top 10 chunks)
2. For each chunk, extract its entities (or look up precomputed entities)
3. For each entity, fetch its top neighbours from the graph
4. Send original chunks + neighbour-entity snippets to the LLM
```

This works well when you have a pre-existing knowledge graph (e.g., a CRM, a product catalog, a Wikipedia dump). Neo4j ships a reference implementation of this pattern.

## When to use which structured pattern

- **Your data is already in a relational DB:** text-to-SQL. Do not over-engineer.
- **Your data is in an API (Jira, Stripe, GitHub):** text-to-tool. Give the LLM a structured API client and let it make calls.
- **Your data is unstructured text and you need global summaries across the whole corpus:** GraphRAG global search.
- **Your data is unstructured text and you need entity-centric questions ("everything about X and its relations"):** GraphRAG local search, or vector RAG + graph neighbourhood expansion.
- **Your data is unstructured text and users mostly ask ordinary lookup questions:** regular RAG. Do not build a graph you don't need.

## Evaluation for structured retrieval

The eval harness from lesson 05 does not quite work for structured retrieval because "the right chunk" is not the right success criterion. Instead:

- **Text-to-SQL:** Check that the generated SQL returns the correct rows. Build eval pairs of `(question, expected_result_rows)` and compare. `sqlglot` can help with query-equivalence checks.
- **GraphRAG:** End-to-end answer quality, judged by an LLM or a human, comparing GraphRAG answers to a reference corpus-level summary.

In both cases you are evaluating the final answer more than the retrieval step, because the "retrieval" is doing structured work that a chunk-level metric cannot capture.

## What to remember

- Vector RAG cannot answer questions about counts, aggregations, relationships, or global corpus summaries. Structured retrieval can.
- Text-to-SQL is the go-to pattern when the data is already in a database. Read-only role, query timeout, row limit, schema card, few-shot examples.
- GraphRAG builds an entity-and-relationship graph from unstructured text, plus community summaries. Local search for entities; global search for corpus-wide summaries.
- GraphRAG indexing is 10–50× more expensive than vanilla RAG. Plan for it.
- Route queries to the right pipeline with a cheap classifier rather than forcing every query through one path.
- A lightweight alternative to full GraphRAG is vector retrieval + graph-neighbourhood expansion on retrieved entities.

## References

- Microsoft, *GraphRAG — A graph-based approach to question-answering over private text corpora*. https://microsoft.github.io/graphrag/
- Edge et al. 2024, *From Local to Global — a Graph RAG Approach to Query-Focused Summarization*. https://arxiv.org/abs/2404.16130
- LlamaIndex, *Text-to-SQL*. https://developers.llamaindex.ai/python/framework/use_cases/agents/text_to_sql/
- Anthropic, *Tool use — database tool patterns*. https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Neo4j, *GraphRAG and knowledge graphs for RAG*. https://neo4j.com/developer-blog/knowledge-graph-rag-application/
- Rajkumar et al. 2022, *Evaluating the text-to-SQL capabilities of large language models*. https://arxiv.org/abs/2204.00498
- `sqlglot` — SQL parser and dialect translator for Python. https://github.com/tobymao/sqlglot
