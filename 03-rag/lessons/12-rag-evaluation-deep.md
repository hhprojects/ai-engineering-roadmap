# 12 — RAG Evaluation in Depth: Faithfulness, Groundedness, Context Precision

> Retrieval metrics tell you the right chunks were found; generation metrics tell you the model used them correctly — and without both, your "evaluation" is not actually evaluating the thing users care about.

Lesson 05 built the baseline: hit@k and MRR to measure whether the retriever found the right chunk. That is a prerequisite for quality but it is not quality. A system with 100% retrieval hit rate can still ship wrong answers if the generator hallucinates, cherry-picks, ignores context, or confabulates confident-sounding fiction from adjacent chunks. This lesson is the deep-evaluation layer — the metrics that catch hallucinations and the frameworks (Ragas, TruLens, LangSmith) that implement them.

## The four dimensions of RAG answer quality

There is a tidy 2×2 matrix that every modern evaluation framework organises itself around:

|              | Quality about the *context* | Quality about the *answer* |
|---|---|---|
| **Precision** | **Context precision:** are the retrieved chunks actually relevant to the question? | **Faithfulness / groundedness:** is the answer fully supported by the retrieved context? |
| **Recall** | **Context recall:** did we retrieve *all* the information needed to answer? | **Answer relevance:** does the answer actually address the user's question? |

Read that table carefully. Each cell catches a distinct failure mode:

- **Low context precision** → the retriever is pulling in noise. Fix: reranker, better chunking, query rewriting.
- **Low context recall** → the retriever is missing information the answer needs. Fix: top-k, hybrid search, HyDE, multi-hop retrieval.
- **Low faithfulness** → the model is saying things the context does not support. Fix: grounding prompt, smaller top-k (less noise), faithfulness-in-the-loop.
- **Low answer relevance** → the model is answering a different question. Fix: prompt, query understanding, decomposition.

You want to measure all four. Ragas, TruLens, and LangSmith give you them by name.

## Faithfulness (groundedness)

The most important RAG-specific metric. "Is every claim in the answer supported by the retrieved context?" If the answer says "Acme's revenue grew 8%" but the context only says "Acme's revenue grew," the answer is unfaithful even if it happens to be true — and the user cannot verify it from the cited sources.

**How Ragas computes faithfulness:**

1. **Claim extraction.** An LLM breaks the answer into atomic factual claims. "The Earth is round and orbits the Sun" → `["The Earth is round", "The Earth orbits the Sun"]`.
2. **Claim verification.** For each claim, the LLM is asked whether the claim is supported by the retrieved context. Binary verdict: supported or not.
3. **Score.** `faithfulness = supported_claims / total_claims`, in `[0, 1]`.

A score of 1.0 means every claim in the answer is grounded. Lower scores point to hallucinations. In practice, faithfulness above 0.9 is achievable with a decent grounding prompt and a reranker; below 0.8 means something is wrong.

**A minimal custom implementation** (for when you want to avoid Ragas as a dependency):

```python
CLAIM_EXTRACT_PROMPT = """
Break the following answer into a list of short, atomic factual claims.
Return one claim per line. Ignore opinions and stylistic language.

Answer: {answer}
"""

CLAIM_VERIFY_PROMPT = """
Given the context and a claim, answer with exactly one word:
SUPPORTED if the claim is directly supported by the context,
UNSUPPORTED otherwise.

Context:
{context}

Claim: {claim}
"""

def faithfulness(answer: str, context: str, judge_model="claude-haiku-4-5") -> float:
    claims = llm_lines(CLAIM_EXTRACT_PROMPT.format(answer=answer), model=judge_model)
    if not claims:
        return 1.0
    supported = sum(
        1 for c in claims
        if llm_word(CLAIM_VERIFY_PROMPT.format(context=context, claim=c), model=judge_model)
        == "SUPPORTED"
    )
    return supported / len(claims)
```

That is 30 lines and does the job. Use Ragas if you want more features; roll your own if you want control.

## Answer relevance

"Does the answer address the user's question, regardless of whether it is correct?" A response that cites three chunks perfectly and then answers a *different* question has high faithfulness and low answer relevance. This happens more than you would expect when the retriever finds almost-matching chunks and the model helpfully answers what it found rather than what was asked.

Ragas computes answer relevance by having the LLM *generate k plausible questions* from the answer, then measuring the embedding similarity between each generated question and the original question. If the answer is relevant to the question, reverse-generated questions will be close to the original; if not, they will drift. Score is the average similarity.

It is a clever metric precisely because it catches the "answered the wrong question" failure that a human reviewer would catch immediately but a faithfulness metric misses.

## Context precision

"Of the chunks we retrieved, which ones were actually useful to answer the question, and how were they ordered?" High context precision means the useful chunks are at the top of the ranking; low context precision means useful chunks are mixed in with noise.

Ragas computes context precision with an LLM judge that, for each retrieved chunk, asks "is this chunk relevant to the question?" and then weights by rank (so chunks at the top count more — basically MAP, Mean Average Precision).

**Why it matters beyond hit@k:** hit@k only checks *if* the right chunk is in the list. Context precision checks where it sits and how much noise surrounds it. Two retrievers can both have hit@5 = 0.9, but if retriever A puts the correct chunk at rank 1 and retriever B puts it at rank 5 buried under four irrelevant chunks, retriever A is much better at protecting the generator from noise.

## Context recall

"Did we retrieve all the information needed to answer?" For a question like "What are the top 5 reasons customers churned?", context recall is 1.0 only if chunks about *all five* reasons were retrieved. A retriever that finds only 3 of the 5 reasons has hit rate = 1.0 (each found reason matches) but context recall = 0.6 — and the model will confidently answer with 3 reasons as if that were the complete list.

Ragas computes context recall by decomposing the **ground-truth answer** into atomic statements and checking how many of them are supported by the retrieved context. This requires labelled reference answers, which is why it is harder to automate than the others.

## LLM-as-judge done properly

All of the above rely on LLM-as-judge — using a separate LLM to score outputs against criteria. LLM-as-judge is powerful but has well-known failure modes:

- **Position bias.** When asked to compare two answers (A vs. B), judges often prefer A. Fix: evaluate each candidate independently, or randomise order and average.
- **Verbosity bias.** Longer answers score higher regardless of quality. Fix: penalise length explicitly in the rubric or cap the answer length.
- **Self-preference.** A judge model often prefers answers from the same model family. Fix: use a different provider's model as the judge (use Claude to judge GPT and vice versa).
- **Calibration drift.** Judge scores are not comparable across model versions. Fix: version-pin your judge and re-baseline when you upgrade.
- **Rubric ambiguity.** "Rate on a scale of 1–5" produces noisy scores; "Is this answer fully supported by the context? yes/no" produces much more reliable ones. Prefer binary or small ordinal scales.

**The judge prompt matters more than the judge model.** A 3-line rubric on Haiku with a clear yes/no question beats a vague rubric on Sonnet every time. Write rubrics like you would write unit tests.

## The Ragas evaluation loop

Ragas is the most popular open-source RAG eval framework in 2026. You feed it a dataset of `(question, answer, contexts, reference_answer)` quadruples and it computes all four metrics plus a few extras (noise sensitivity, answer correctness). The minimal loop:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

samples = [
    {
        "question": q["query"],
        "answer": my_rag_pipeline(q["query"]),
        "contexts": [c["text"] for c in retrieve(q["query"])],
        "ground_truth": q["reference_answer"],
    }
    for q in eval_set
]

result = evaluate(
    Dataset.from_list(samples),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)
# {'faithfulness': 0.91, 'answer_relevancy': 0.87,
#  'context_precision': 0.82, 'context_recall': 0.79}
```

Ragas handles claim extraction, LLM-as-judge orchestration, and concurrency. The tradeoff: it is LLM-call-heavy. A 100-question eval with all four metrics is on the order of 600–1000 LLM calls and a few dollars in inference. Batch it, cache results, and run it periodically — nightly or per PR.

Alternatives worth knowing:

- **TruLens** — similar metrics, better observability integration, ties into LangChain natively.
- **LangSmith evaluations** — hosted, integrates with the LangChain ecosystem, adds trace-level diffing.
- **Phoenix / Arize** — production-grade observability with built-in eval templates.
- **DeepEval** — pytest-style assertions (`assert_faithfulness(answer, context, threshold=0.9)`).

For new projects I would start with Ragas for the metrics and Langfuse or Phoenix for tracing. The mix is less important than the discipline of actually running the evaluation on every non-trivial change.

## Online vs. offline evaluation

Everything so far is **offline** — you build a test set and score your system on it. Offline eval catches regressions and lets you compare versions, but it cannot tell you whether real users are happy.

**Online evaluation** runs metrics on production traffic in real time:

- **Per-query scoring.** When a user sends a query, in the background score the response with faithfulness and answer relevance. Flag low-scoring responses for review.
- **Thumbs-up / thumbs-down feedback** wired into the UI. Log the query, retrieved contexts, and answer, grouped by feedback label. Use the negative-labelled traces to seed offline eval and improve the test set.
- **Alerting.** Average faithfulness drops below threshold → page an engineer. Context precision drops after a deploy → auto-rollback.

Jason Liu's Level 3 ("observability") and Level 5 ("understanding shortcomings") from lesson 01 both live here. The eval harness is the measuring instrument; online evaluation is the measuring in production.

## A realistic RAG eval scorecard

Here is what a real production scorecard might look like after you wire everything up:

| Metric | Score | Target | Status |
|---|---|---|---|
| hit@5 | 0.91 | ≥ 0.90 | ✅ |
| MRR | 0.74 | ≥ 0.70 | ✅ |
| Faithfulness | 0.93 | ≥ 0.90 | ✅ |
| Answer relevance | 0.88 | ≥ 0.85 | ✅ |
| Context precision | 0.81 | ≥ 0.80 | ✅ |
| Context recall | 0.77 | ≥ 0.80 | ⚠️  |
| Thumbs-up rate (7 days) | 0.84 | ≥ 0.80 | ✅ |
| P95 latency | 2.4s | ≤ 3s | ✅ |

The one warning is context recall — the retriever is missing some of the information needed for full answers. The fix is probably to increase top-k or add a second retrieval round. The other metrics are green, which tells you *exactly* where to spend your optimisation budget.

Without a scorecard like this, you are guessing.

## What to remember

- Retrieval metrics are necessary but not sufficient. Measure generation quality too.
- Faithfulness / groundedness, answer relevance, context precision, context recall — the four dimensions Ragas organises around.
- Faithfulness (fraction of claims supported by context) is the single most important generation metric. Always measure it.
- LLM-as-judge works when the rubric is binary, the judge is model-pinned, and position bias is controlled.
- Ragas is the default open-source eval framework. TruLens, LangSmith, and DeepEval are valid alternatives.
- Offline eval catches regressions. Online eval (per-query scoring, user feedback) catches reality.
- Build a scorecard with thresholds. Green/yellow/red status tells you where to focus optimisation.

## References

- Ragas documentation — available metrics. https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- Ragas GitHub — source and examples. https://github.com/explodinggradients/ragas
- Shahul Es et al. 2023, *Ragas — Automated Evaluation of Retrieval Augmented Generation*. https://arxiv.org/abs/2309.15217
- TruLens documentation. https://www.trulens.org/
- LangSmith, *Dataset and evaluation*. https://docs.smith.langchain.com/evaluation
- Hamel Husain, *Your AI product needs evals*. https://hamel.dev/blog/posts/evals/
- Zheng et al. 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. https://arxiv.org/abs/2306.05685
- Phoenix by Arize — open-source LLM observability. https://github.com/Arize-ai/phoenix
