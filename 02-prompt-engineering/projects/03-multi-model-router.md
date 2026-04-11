# Project 3 — Multi-Model Router

🟠 **Advanced** · ~10-14 hours · ~$3-6 in API credits

Not every query needs Claude Opus 4.6. The biggest single cost optimization in production LLM apps is routing — sending simple queries to cheap models and hard queries to premium ones. In this project you'll build a routing system that classifies incoming queries, dispatches them to the right tier, and measures the cost-vs-quality trade-off with a proper evaluation framework. When you're done, you'll have the architecture that every AI company in production uses (and the evidence to prove it works).

---

## Prerequisites

- Finished **Lessons 7, 8, 11, 12** and Module 1 Lesson 13 (role prompting, structured outputs, evals, model selection)
- Finished **Projects 1 and 2** — you'll reuse the provider abstraction and extractor patterns
- Python 3.11+
- API keys for at least OpenAI, Anthropic, and Groq (for cheap Llama-based routing)

---

## What you'll build

A Python library + CLI that:

1. **Classifies** incoming queries by type and complexity using a cheap, fast model.
2. **Routes** each query to an appropriate tier (cheap/medium/premium) based on classification and configurable rules.
3. **Logs** every routing decision: query, classification, chosen model, response, latency, cost.
4. **Evaluates** routing against a gold-standard "always use flagship" baseline, measuring both cost savings and quality gap.
5. **Falls back** when the routed model fails or returns low-confidence output, escalating to the next tier.
6. **Config-driven rules** via YAML so you can adjust routing without code changes.
7. **Cost dashboard** showing cumulative spend, per-tier breakdowns, and savings vs. baseline.
8. Runs on an **eval set of ≥50 queries** spanning the full difficulty range.

This is the most realistic production-LLM project in the module. The patterns you build here show up unchanged in real AI products.

---

## What you'll learn

- Query classification with a small-model router
- Multi-tier architecture (cheap → medium → premium)
- Cost vs. quality evaluation — the real trade-off of production LLM work
- Fallback and escalation patterns
- Config-driven routing rules
- Building a baseline to measure savings against
- The discipline of shipping "good enough at a fraction of the cost"

---

## Tech stack

- **Python 3.11+**
- `openai`, `anthropic`, `groq` — provider SDKs
- `instructor` — for the classifier (reusing Project 2's pattern)
- `litellm` — optional, for a unified multi-provider interface
- `typer` + `rich` — CLI (reusing Project 1)
- `pyyaml` — routing config
- `pydantic` — typed configs and records
- `sqlite3` (stdlib) — routing log
- `pytest` — tests

---

## Setup

```bash
mkdir model-router && cd model-router
python -m venv venv
source venv/bin/activate
pip install openai anthropic groq instructor typer rich pyyaml pydantic python-dotenv pytest
```

```
model-router/
├── pyproject.toml
├── router/
│   ├── __init__.py
│   ├── cli.py
│   ├── classifier.py      ← classifies queries
│   ├── router.py          ← routing logic
│   ├── tiers.py           ← tier definitions and config loading
│   ├── providers.py       ← reused from Project 1
│   ├── store.py           ← SQLite log
│   └── evaluator.py       ← compares routed vs baseline
├── config/
│   └── routing.yaml       ← the routing rules
├── eval_data/
│   ├── queries.jsonl      ← 50+ test queries
│   └── rubric.md          ← how to judge quality
├── tests/
└── README.md
```

---

## Requirements

### Must have

#### Tier definitions

```yaml
# config/routing.yaml
tiers:
  cheap:
    provider: groq
    model: llama-3.3-70b-versatile
    cost_input_per_mtok: 0.59
    cost_output_per_mtok: 0.79

  medium:
    provider: anthropic
    model: claude-haiku-4-5
    cost_input_per_mtok: 1.00
    cost_output_per_mtok: 5.00

  premium:
    provider: anthropic
    model: claude-sonnet-4-6
    cost_input_per_mtok: 3.00
    cost_output_per_mtok: 15.00

classifier:
  provider: groq
  model: llama-3.3-70b-versatile

routing_rules:
  - category: simple_factual
    tier: cheap
  - category: general_qa
    tier: cheap
  - category: creative_writing
    tier: medium
  - category: code_generation
    tier: medium
  - category: complex_reasoning
    tier: premium
  - category: nuanced_analysis
    tier: premium
  - category: unclassified         # the escape hatch
    tier: premium

fallback:
  enabled: true
  confidence_threshold: 0.7       # below this, escalate to next tier
  max_escalations: 1              # don't infinite-loop
```

One place to change pricing, one place to change tier mapping. Don't hardcode any of this in Python.

#### Classifier

A small, cheap LLM call that extracts `category` and `confidence` for each query. Use Instructor for reliable structured output.

```python
# router/classifier.py
from pydantic import BaseModel, Field
from typing import Literal
import instructor

Category = Literal[
    "simple_factual",
    "general_qa",
    "creative_writing",
    "code_generation",
    "complex_reasoning",
    "nuanced_analysis",
    "unclassified",
]

class Classification(BaseModel):
    category: Category = Field(
        description=(
            "The type of query. Use 'unclassified' if genuinely unclear. "
            "Definitions: "
            "simple_factual = single-fact lookup (e.g. 'what's the capital of France'). "
            "general_qa = common questions with well-known answers. "
            "creative_writing = poems, stories, ideation. "
            "code_generation = write, debug, or explain code. "
            "complex_reasoning = multi-step logic, math, analysis. "
            "nuanced_analysis = tasks needing domain judgment or careful tone."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are in the classification, 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="One-sentence rationale for the classification"
    )

def classify(query: str, client) -> Classification:
    system = (
        "You are a query classifier for an LLM routing system. "
        "Read each query carefully and categorize it. When unsure, use "
        "'unclassified' and low confidence rather than guessing."
    )
    return client.create(
        response_model=Classification,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    )
```

Test this classifier on its own eval set (10-20 labeled queries) before building the router on top of it. If classification accuracy is poor, your whole routing system will be poor — no amount of routing logic rescues a bad classifier.

#### Router

```python
# router/router.py
from pydantic import BaseModel
from datetime import datetime
from router.classifier import classify, Classification
from router.tiers import load_config, Tier
from router.providers import call_provider
import uuid

class RoutedResponse(BaseModel):
    query: str
    classification: Classification
    tier_used: str
    escalated: bool
    response: str
    latency_ms: float
    cost_usd: float
    timestamp: datetime

def route(query: str) -> RoutedResponse:
    config = load_config()
    classifier_client = get_classifier_client(config)

    # Step 1: classify
    classification = classify(query, classifier_client)

    # Step 2: pick tier (with confidence check)
    chosen_tier_name = map_category_to_tier(classification.category, config)
    if config["fallback"]["enabled"]:
        if classification.confidence < config["fallback"]["confidence_threshold"]:
            chosen_tier_name = escalate(chosen_tier_name, config)

    chosen_tier = config["tiers"][chosen_tier_name]

    # Step 3: call chosen tier
    result = call_provider(query, chosen_tier)

    return RoutedResponse(
        query=query,
        classification=classification,
        tier_used=chosen_tier_name,
        escalated=(chosen_tier_name != map_category_to_tier(classification.category, config)),
        response=result.text,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd + classification_cost(classifier_client, query),
        timestamp=datetime.utcnow(),
    )
```

Note: the total cost includes the classifier call. Don't hide it — that call is real money and needs to count.

#### Routing log

Every call to `route()` writes a row to SQLite. Columns: timestamp, query (hashed for privacy if you want), category, confidence, tier, escalated, response, latency, classifier_cost, response_cost, total_cost.

This log is what you query for metrics. Don't build a fancy dashboard — a few SQL queries will do.

#### Eval set

Build a JSONL of ≥50 queries spanning all categories. For each, include:

```json
{
  "query": "What's the square root of 196?",
  "expected_category": "simple_factual",
  "expected_tier": "cheap",
  "quality_rubric": "Answer should be exactly 14, nothing more",
  "min_acceptable_quality": 0.9
}
```

Distribution of categories matters. Roughly:

- 30% `simple_factual` + `general_qa` (should go to cheap)
- 30% `creative_writing` + `code_generation` (should go to medium)
- 30% `complex_reasoning` + `nuanced_analysis` (should go to premium)
- 10% adversarial: queries designed to confuse the classifier, or edge cases

#### Baseline comparison

Build an `evaluator.py` that:

1. Runs each query through `route()` → records tier, response, cost
2. Runs each query through a **baseline** — always using the premium tier — and records response, cost
3. Uses an **LLM-as-judge** (a strong model, different from the router and baseline) to score both responses on a 1-5 scale against the rubric
4. Computes:
   - Total cost of routed pipeline
   - Total cost of baseline pipeline
   - Cost savings (%)
   - Average quality score — routed vs baseline
   - Quality gap (how much quality did you lose?)
   - Per-tier breakdowns (how many queries went to each tier)
   - Escalation rate (how often the confidence fallback fired)

The expected result of a well-tuned router is something like: **~70% cost reduction, ~5% quality drop.** You're trading a little quality for a lot of money. Your write-up should make that trade-off explicit.

#### CLI

```bash
# Route a single query
router ask "What's 2+2?"

# Route a query and show the classification and reasoning
router ask "Review this code..." --verbose

# Run the full eval
router eval --output results.json

# Show cost dashboard
router stats --since 7d

# Set a per-user override
router override --user alice always-premium
```

### Stretch goals (pick ≥1)

- **Semantic caching.** Cache responses for queries that embed similarly to previous queries. Skip the LLM entirely on near-duplicates. Massive savings for repeat queries.
- **Rule-based pre-routing.** For certain query patterns (short queries, keyword matches, length thresholds), skip the LLM classifier and route directly with heuristics. Faster and cheaper.
- **Per-user routing policies.** Some users (VIPs, enterprise customers) always go to premium regardless of classification. Config-driven.
- **Rolling eval.** Automatically re-run the eval nightly against the latest production queries. Alert if quality drops or cost drifts upward.
- **Dynamic confidence threshold.** Instead of a fixed 0.7 threshold, tune it based on the cost-quality budget you want. Higher threshold → more escalations, higher cost, better quality.
- **Budget enforcement.** Given a $X/day budget, refuse to escalate once the budget is half-spent on queries that normally would escalate. This lets you ship with cost predictability.
- **Multi-LLM ensemble on hard queries.** For the `complex_reasoning` tier, run two premium models in parallel and pick the consensus. Expensive but effective.

---

## Evaluation rubric

- [ ] Router classifies + routes + logs every query
- [ ] All routing rules are in YAML, not Python
- [ ] Classifier has its own ≥20-query eval and achieves ≥80% accuracy
- [ ] Confidence-based fallback works: low-confidence classifications escalate to a higher tier
- [ ] Log in SQLite captures every relevant field for later analysis
- [ ] Eval set of ≥50 queries, labeled with expected tier and quality rubric
- [ ] Baseline comparison runs every query through both router and "always premium"
- [ ] LLM-as-judge scores both pipelines on quality, 1-5 scale
- [ ] Reported metrics: cost savings %, quality gap, per-tier distribution, escalation rate
- [ ] Cost savings ≥50% with quality drop ≤10% on your eval set
- [ ] At least one stretch goal done
- [ ] README documents your routing decisions, eval results, and the cost-quality trade-off you chose

---

## Common pitfalls

- **Building the router before the classifier is validated.** If your classifier gets 60% accuracy, your routing will fail regardless of how good the rest of the code is. Validate the classifier on its own eval first.
- **Using an LLM classifier when heuristics would work.** If your queries fall into obvious categories by keyword or length, a Python function is faster, cheaper, and more reliable than an LLM call. Use the LLM only for genuinely fuzzy cases.
- **Hardcoding the routing rules.** The whole point is to iterate on routing. Keep it in YAML from day one.
- **Not counting the classifier cost.** The classifier is an extra LLM call per query. For cheap queries, the classifier can dominate the total cost. If you're routing to a $0.001 call and paying $0.0005 to classify, the savings are smaller than they look.
- **Measuring on a biased eval set.** If 90% of your eval is easy queries, routing will look great (cheap tier handles almost everything). If 90% is hard queries, routing will look pointless. Match the distribution to what your production queries actually look like.
- **Using the same model for router and baseline.** You'll discover your router is "as good as" the baseline because it *is* the baseline. Use different models for the premium tier and the judge.
- **Not handling classifier failures.** What happens when the LLM classifier fails (rate limit, timeout, weird output)? Default to the premium tier. Never fail-open to the cheap tier — that's how you ship wrong answers at scale.
- **Routing based on the prompt, not the full context.** If your system uses long context (prior conversation, retrieved docs), the classifier needs to see all of it to judge correctly. Or — cheaper — classify based on the latest user turn only, if that's usually indicative.
- **Shipping without an override mechanism.** Someone will need to force a query to a specific tier for debugging. Build the override early.
- **Believing your LLM-as-judge.** Judge results are noisy. Spot-check 10% of them against your own judgment before trusting the aggregate numbers.

---

## Cost estimate

| Activity | Approximate cost |
|---|---|
| Classifier tuning (lots of cheap calls) | ~$0.50 |
| Router iteration on eval set | ~$1-2 |
| Full eval run (routed × 50 + baseline × 50 + judge × 100) | ~$2-3 |
| Stretch goals | ~$1 |
| **Total** | **~$4-6** |

Use Groq's free tier aggressively for the classifier. Reserve paid flagship calls for the baseline and the judge.

---

## What to deliver

```
model-router/
├── pyproject.toml
├── router/
├── config/
│   └── routing.yaml
├── eval_data/
│   ├── queries.jsonl
│   └── rubric.md
├── tests/
├── results/
│   ├── eval_results.json    ← your final numbers
│   └── plots/                ← optional cost-vs-quality scatter
├── README.md
└── .env.example
```

README must include:
- Architecture diagram (routing + classification + fallback)
- The routing rules you chose and why
- Eval results: cost savings, quality gap, per-tier breakdown, escalation rate
- A 1-paragraph recommendation: "Based on these results, I would ship this router because..."
- Known limitations and what you'd do differently

---

## Going further (after you finish)

- **Integrate it into a real app.** Take Project 1 (prompt playground) or Project 2 (structured extractor) and run every call through the router. Measure real-world savings.
- **Tune routing with real production data.** Once you have a week of routing logs, re-tune the classifier against actual queries. Production data is always more informative than synthetic.
- **Add prompt caching on top** (Module 1 Lesson 11). Routing + caching is the canonical production cost-optimization stack.
- **Replace the LLM classifier with a fine-tuned small model.** Train a tiny classifier (Phi-3, Gemma-2, distilBERT) on your labeled data. Faster, cheaper, more predictable than an LLM.
- **Open source it.** A good router is a legitimate pre-built component teams can adopt. Ship it as a PyPI package with configurable tiers and providers.

---

## References

- Module 1 Lesson 13, *Choosing a Model*. https://github.com/hhprojects/ai-engineering-roadmap (your own previous work)
- LiteLLM, *Unified LLM interface*. https://docs.litellm.ai/
- Hamel Husain, *Eval frameworks*. https://hamel.dev/blog/posts/evals/
- Artificial Analysis, *LLM benchmarks and pricing*. https://artificialanalysis.ai/
- FrugalGPT paper (Chen et al., 2023), *How to Use Large Language Models While Reducing Cost*. https://arxiv.org/abs/2305.05176
- OpenRouter, *LLM routing at scale*. https://openrouter.ai/
- Anthropic, *Comparing Claude models for routing decisions*. https://docs.claude.com/en/docs/about-claude/models/overview

---

[← Previous Project](02-structured-extractor.md) | [Back to Prompt Engineering](../README.md)
