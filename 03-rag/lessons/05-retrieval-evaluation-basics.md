# 05 — Retrieval Quality and Evaluation Basics

> If you cannot measure your retrieval quality, every "improvement" you make is superstition; this lesson is about the smallest possible eval harness that turns RAG development from vibes into engineering.

You should build an eval harness on day two of a RAG project. Not day thirty. Not after "we have enough data." Day two. The reason is simple: everything between here and production is a series of trade-offs between chunking, embedding model, top-k, hybrid weights, reranker, and query rewriting, and without a measurement you cannot tell whether any of them is helping. Teams that skip evaluation end up chasing their tails for months and shipping systems they cannot confidently improve.

This lesson is a baseline — just enough evaluation to compare configurations and catch regressions. Deep evaluation (faithfulness, groundedness, context precision, LLM-as-judge) is lesson 12. Start here.

## The two questions an eval harness answers

Every RAG evaluation boils down to two orthogonal questions:

1. **Did we retrieve the right stuff?** (retrieval quality)
2. **Did the LLM answer the question correctly given the retrieved stuff?** (generation quality)

You must measure these *separately*. If your end-to-end answer is wrong, it matters whether the retriever failed to find the answer at all (retrieval bug, fix the index) or whether the retriever found the answer and the generator ignored it (prompt bug, fix the grounding prompt). Teams that only look at end-to-end "the answer is wrong" spend weeks tuning the wrong knob.

This lesson is about the first question. Lesson 12 covers the second.

## Building your first eval set

The minimum viable eval set is **30 to 50 question/ground-truth pairs**. A pair consists of:

- The **query** a user would ask.
- The **expected document** (or specific chunk) that contains the answer.

Yes, 30 questions feels tiny. Yes, it is enough to catch most real regressions. You are not building a benchmark paper; you are building an instrument that tells you "is version B of my retrieval pipeline better or worse than version A on the queries I care about?"

**Where the questions come from:**

- **You write them.** Read a sample of the corpus and invent plausible questions. This is faster than it sounds — 20 questions per hour is normal.
- **Synthetic generation with an LLM.** Feed each chunk (or each document section) to a cheap model and ask it to generate 2–3 questions whose answer is in that chunk. Ragas has a `TestsetGenerator` that does this for you. The quality is mediocre but you get 200 questions in ten minutes, and you can hand-curate the best 50.
- **Production logs.** Once you have users, their actual queries are by far the best eval set. Sample them, label the correct source manually, and replace your synthetic eval questions as you go.
- **Customer support tickets, FAQ, internal Slack.** Gold mines of real questions about the corpus.

**Good eval questions are diverse.** Mix factual lookups ("What was the revenue in Q3?"), conceptual questions ("How does reranking work?"), multi-hop questions ("What did X say about Y in the context of Z?"), and negative examples ("Does the manual cover feature X?" where the answer is "no"). If all your questions are "find the one sentence" lookups, your eval will show that the naive pipeline is fine and you will miss the structural failures.

**Ground truth is a document ID, not an answer string.** At the retrieval stage, you are checking "did the right chunk show up in the top-k?" not "did the final answer match?" The question of whether the final answer is correct is a generation-quality question (lesson 12) and is harder to automate.

Store the eval set as a plain file — JSON or CSV — under version control next to the code:

```json
[
  {
    "id": "q001",
    "query": "What is the default HNSW m parameter in pgvector?",
    "expected_chunk_ids": ["chunks-pgvector-hnsw-params"],
    "notes": "Factual lookup. Answer should mention m=16."
  },
  ...
]
```

## The three metrics you need

For retrieval quality, three metrics cover almost everything in practice:

### Hit@k

The fraction of queries where the expected chunk appears anywhere in the top-k results.

```python
def hit_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> int:
    return int(any(i in expected_ids for i in retrieved_ids[:k]))
```

Averaged across the eval set, hit@5 is your headline retrieval number. It answers: "In what fraction of queries does the model even have a chance to answer correctly?"

**Targets:** a hit@5 above 90% means the retriever is not your bottleneck. Below 70% means the retriever is why the system fails; fix chunking and hybrid search before anything else.

### MRR (Mean Reciprocal Rank)

The average of `1 / rank` across all queries, where `rank` is the position (1-indexed) of the first correct document in the retrieved list. If the correct document is not retrieved at all, the contribution is 0.

```python
def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_ids:
            return 1.0 / rank
    return 0.0
```

MRR captures "how near the top is the first correct result?" It is more sensitive than hit@k — a retriever that nudges the correct result from rank 5 to rank 2 gains MRR without changing hit@5. That makes it the right metric when you are tuning reranker or hybrid weights.

**Targets:** MRR above 0.7 is "most answers are in the top 2" territory and tends to correlate with good user experience.

### nDCG@k (Normalised Discounted Cumulative Gain)

A more sophisticated metric from the information retrieval world. It weighs each correct result by `1 / log2(rank + 1)` so higher positions count more, then normalises against the best possible ranking. Use it when you have multiple correct documents per query with different relevance levels (e.g., "highly relevant" vs. "somewhat relevant"), which you rarely do in RAG.

For most RAG projects, **hit@5 and MRR are enough**. Do not overcomplicate.

## A minimal eval runner

The eval loop is a for-loop. There is no framework required until there is:

```python
import json
from pathlib import Path

def run_eval(eval_set: list[dict], retriever, k: int = 5) -> dict:
    hits = 0
    rrs = []
    per_query = []
    for item in eval_set:
        retrieved = retriever.retrieve(item["query"], k=k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        hit = any(cid in item["expected_chunk_ids"] for cid in retrieved_ids)
        hits += int(hit)
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in item["expected_chunk_ids"]:
                rr = 1.0 / rank
                break
        rrs.append(rr)
        per_query.append({
            "id": item["id"],
            "hit": hit,
            "rr": rr,
            "retrieved": retrieved_ids,
        })
    n = len(eval_set)
    return {
        "hit_at_k": hits / n,
        "mrr": sum(rrs) / n,
        "k": k,
        "n": n,
        "per_query": per_query,
    }

eval_set = json.loads(Path("eval/questions.json").read_text())
results = run_eval(eval_set, my_retriever, k=5)
print(f"hit@5 = {results['hit_at_k']:.2%}  MRR = {results['mrr']:.3f}")
```

That is a complete eval harness. Run it before and after every change that touches retrieval — chunk size, embedding model, reranker weights, top-k. Commit the output alongside the change. If a change drops hit@5 by more than a percentage point or two, investigate before merging.

## The failure bucket — where the real learning happens

Aggregate metrics lie. A retriever that improves from 80% to 85% hit@5 is genuinely 5% better *on average*, but you also just gained and lost queries along the way, and the *which* queries matters more than the average.

Every eval run should produce a **failure bucket**: the list of queries that failed in this run. Look at them. For each failure, diagnose:

- **Missing from index.** The correct chunk does not exist. Fix the ingestion.
- **In the index but not retrieved.** The embedding missed it. Try hybrid search, query rewriting, or a different embedder.
- **Retrieved but at low rank.** Try reranking or tuning fusion weights.
- **Ambiguous query.** The query genuinely maps to multiple documents; the ground truth is wrong or incomplete. Update the eval set.

You will find bugs in your eval set as often as you find bugs in your retriever. That is fine — keep updating.

## Measuring the improvement from each layer

A good habit: maintain a spreadsheet (or a small script) that tracks hit@5 and MRR per configuration:

| Config | Description | hit@5 | MRR | Δ |
|---|---|---|---|---|
| v0 | Fixed 500-token chunks, `bge-small-en` | 0.62 | 0.41 | baseline |
| v1 | Recursive 800-token chunks, 100 overlap | 0.71 | 0.48 | +9 / +0.07 |
| v2 | v1 + hybrid (BM25 + dense, RRF) | 0.81 | 0.56 | +10 / +0.08 |
| v3 | v2 + Cohere Rerank top 20 → top 5 | 0.89 | 0.72 | +8 / +0.16 |
| v4 | v3 + HyDE query rewriting | 0.91 | 0.74 | +2 / +0.02 |

This table is the only artifact you need to justify your stack to a skeptical stakeholder. It also tells you when to stop — look at v4: query rewriting added 2 percentage points for a 50% latency increase. You can skip it until later.

## Eval-in-CI

Once the eval set is stable and you have a runner, put it in CI. Every PR that touches retrieval runs the eval and prints the delta against the main branch. If hit@5 drops below a threshold (say, 85% of the main-branch number), the PR is blocked. This catches regressions before they ship and does more for your quality than any amount of code review.

Langfuse, LangSmith, and Ragas all have hosted eval-in-CI support; your own Python script works too. The machinery is less important than the discipline of actually running it.

## Common mistakes

- **Only looking at end-to-end answer quality.** You cannot tell if the retrieval or the generator is broken. Measure them separately.
- **Tuning on the eval set directly.** If you change the retriever until hit@5 on your eval set is 100%, you have overfitted. Hold out 20% of the eval set as a test set you only look at before releases.
- **Metrics without a failure bucket.** You need the actual list of failing queries to learn anything. Print it.
- **Eval set never updates.** The real world changes; your eval should too. Replace 10% of the questions with production logs every month.
- **Eval set that is too easy or too hard.** If every change leaves hit@5 at 100%, the eval is not discriminating. If every change leaves it at 20%, the eval is not achievable. Tune the difficulty to your current system.
- **Comparing runs without fixed seeds.** LLM-as-judge and synthetic generation vary across runs; set seeds and temperatures to 0 where possible.

## What to remember

- Build an eval set on day two. 30–50 question/expected-chunk pairs is enough.
- Measure retrieval quality separately from generation quality. They have different failure modes and different fixes.
- hit@k and MRR are the two metrics you actually need. Skip nDCG unless you have graded relevance.
- The failure bucket is more valuable than the aggregate metric. Always inspect failed queries.
- Track every config change in a table. Know exactly what each layer bought you in quality.
- Put the eval in CI. Regressions should block merges.
- Keep a held-out test set to avoid overfitting.

## References

- Ragas documentation — open-source eval framework for RAG. https://docs.ragas.io/
- Hamel Husain, *Your AI product needs evals*. https://hamel.dev/blog/posts/evals/
- Eugene Yan, *Evaluation & hallucination detection for LLMs*. https://eugeneyan.com/writing/llm-patterns/
- Jason Liu, *Levels of Complexity — evaluations section*. https://jxnl.co/writing/2024/02/28/levels-of-complexity-rag-applications/
- Langfuse, *Evals and experiments*. https://langfuse.com/docs/scores/model-based-evals
- LangSmith, *Dataset and evaluation*. https://docs.smith.langchain.com/evaluation
- Radlinski & Craswell 2017, *A theoretical framework for conversational search* — mental model for IR metrics. https://dl.acm.org/doi/10.1145/3020165.3020183
