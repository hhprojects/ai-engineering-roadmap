# Project 2 — Structured Data Extractor

🟡 **Intermediate** · ~6-8 hours · ~$1-3 in API credits

LLMs are absurdly good at extracting structured data from messy text — if you ask them correctly. In this project you'll build a tool that does it properly: schemas with escape hatches, retries with validator feedback, handles ambiguous inputs without hallucinating, and runs on multiple providers interchangeably. By the end you'll have the scaffolding to extract *anything* — receipts, job listings, contact cards, medical notes — with a two-line code change.

---

## Prerequisites

- Finished **Lessons 3, 4, 8, 12** (anatomy, few-shot, structured outputs, evals)
- Finished Module 1 Lesson 10 (structured outputs overview)
- Python 3.11+
- Comfortable with Pydantic basics
- API keys for OpenAI and Anthropic (at least one of each family)

---

## What you'll build

A Python library + CLI that:

1. Defines **Pydantic models for ≥3 document types** — e.g., receipts, job listings, contact info.
2. Extracts structured data from unstructured text input via **Instructor** (or direct strict-mode) with automatic validation and retry.
3. Handles **partial extractions** gracefully — returns nulls instead of hallucinations, flags missing required fields.
4. Uses **custom Pydantic validators** for rules the schema alone can't express (regex formats, allowed enum values, length constraints).
5. Supports **batch mode** to process a directory of files and emit JSONL.
6. Tracks **per-document success rate** and **cost** across runs.
7. Runs against **≥2 providers** (OpenAI + Anthropic) so you can compare which extracts better for your data.
8. Includes an **eval set** of ≥20 documents per type with known-good extractions, and a script to measure accuracy.

This is the baseline pattern for any LLM-powered data pipeline you'll build in your career. Do it carefully and it'll generalize to dozens of future use cases.

---

## What you'll learn

- Designing Pydantic schemas for LLM extraction (one thing per field, escape hatches, enums)
- Using Instructor for cross-provider structured outputs with retries
- Writing custom validators that re-prompt the model on failure
- Building eval sets and measuring extraction accuracy
- Handling partial / uncertain extractions without hallucination
- Cost and accuracy trade-offs across providers and models
- Reusable extraction pipelines — the pattern, not just one instance

---

## Tech stack

- **Python 3.11+**
- `instructor` — the structured outputs library (Lesson 8)
- `openai`, `anthropic` — provider SDKs
- `pydantic` v2 — schemas and validators
- `typer` — CLI (reused from Project 1)
- `pytest` — eval tests
- `rich` — output tables
- `jsonlines` — batch output

---

## Setup

```bash
mkdir structured-extractor && cd structured-extractor
python -m venv venv
source venv/bin/activate
pip install instructor openai anthropic pydantic typer rich jsonlines pytest python-dotenv
```

```
structured-extractor/
├── pyproject.toml
├── extractor/
│   ├── __init__.py
│   ├── cli.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── receipt.py
│   │   ├── job_listing.py
│   │   └── contact.py
│   ├── extractor.py      ← the core extraction function
│   ├── metrics.py        ← accuracy + cost tracking
│   └── store.py
├── eval_data/
│   ├── receipts/
│   │   ├── sample_01.txt
│   │   ├── sample_01.json     ← expected output
│   │   ├── ...
│   ├── job_listings/
│   └── contacts/
├── tests/
│   └── test_extraction.py
├── .env.example
└── README.md
```

---

## Requirements

### Must have

#### Schema 1 — Receipt

```python
# extractor/schemas/receipt.py
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import date

class LineItem(BaseModel):
    name: str = Field(description="Product or service name as shown on the receipt")
    quantity: float | None = Field(description="Quantity, or None if unclear")
    unit_price: float | None = Field(description="Price per unit in the receipt's currency")
    total: float = Field(description="Total for this line item")

class Receipt(BaseModel):
    merchant_name: str = Field(description="Name of the business")
    merchant_address: str | None = Field(description="Street address, None if not present")
    transaction_date: str | None = Field(
        description="Transaction date in YYYY-MM-DD format, None if not legible"
    )
    total: float = Field(description="Final total paid, including tax")
    currency: Literal["USD", "SGD", "EUR", "GBP", "JPY", "CNY", "unknown"] = Field(
        description="Currency code. Use 'unknown' if no currency is explicit."
    )
    line_items: list[LineItem] = Field(
        description="Individual purchased items. Empty list if not itemized."
    )
    tax_amount: float | None = Field(
        description="Total tax charged, or None if not shown separately"
    )

    @field_validator("transaction_date")
    @classmethod
    def validate_iso_date(cls, v):
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("transaction_date must be ISO format YYYY-MM-DD")
        return v

    @field_validator("total")
    @classmethod
    def total_positive(cls, v):
        if v < 0:
            raise ValueError("total must be non-negative")
        return v
```

Notice every field has an `Optional` escape hatch (except `merchant_name` and `total`, which every receipt has). Validators enforce ISO dates and non-negative totals. The enum for `currency` includes `"unknown"`.

#### Schema 2 — Job listing

```python
# extractor/schemas/job_listing.py
from typing import Literal
from pydantic import BaseModel, Field

Seniority = Literal["intern", "junior", "mid", "senior", "staff", "principal", "unknown"]
WorkMode = Literal["remote", "onsite", "hybrid", "unknown"]

class SalaryRange(BaseModel):
    min_amount: float | None = Field(description="Minimum salary, None if not stated")
    max_amount: float | None = Field(description="Maximum salary, None if not stated")
    currency: Literal["USD", "SGD", "EUR", "GBP", "unknown"] = "unknown"
    period: Literal["year", "month", "hour", "unknown"] = "unknown"

class JobListing(BaseModel):
    title: str = Field(description="Job title as posted")
    company: str | None = Field(description="Company name, None if not mentioned")
    location: str | None = Field(description="City and country, None if fully remote or unstated")
    seniority: Seniority = Field(
        description="Seniority level. 'unknown' if the listing doesn't say."
    )
    work_mode: WorkMode = Field(description="Work arrangement")
    salary: SalaryRange | None = Field(
        description="Salary range, None if not stated in the listing"
    )
    required_skills: list[str] = Field(
        description="Skills/technologies explicitly required. Empty if none listed."
    )
    nice_to_have: list[str] = Field(
        description="Skills listed as 'preferred' or 'nice to have'. Empty if none."
    )
    years_experience_min: int | None = Field(
        description="Minimum years of experience required, None if unstated"
    )
```

#### Schema 3 — Contact info

```python
# extractor/schemas/contact.py
from pydantic import BaseModel, Field, field_validator
import re

class Contact(BaseModel):
    full_name: str = Field(description="Full name of the person")
    first_name: str | None = Field(description="First name separated from full name")
    last_name: str | None = Field(description="Last name separated from full name")
    email: str | None = Field(description="Email address, None if not present")
    phone: str | None = Field(
        description="Phone number in E.164 format (e.g., +6591234567)"
    )
    company: str | None = Field(description="Company or organization")
    title: str | None = Field(description="Job title")
    linkedin_url: str | None = Field(description="LinkedIn profile URL if present")

    @field_validator("email")
    @classmethod
    def valid_email(cls, v):
        if v is None:
            return v
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError(f"Invalid email format: {v}")
        return v

    @field_validator("phone")
    @classmethod
    def e164_format(cls, v):
        if v is None:
            return v
        if not re.match(r"^\+\d{7,15}$", v):
            raise ValueError(
                f"Phone must be E.164 format starting with +. Got: {v}. "
                "Normalize the phone number by removing spaces and dashes."
            )
        return v
```

The regex validators will reject malformed outputs and trigger Instructor's retry. The second retry often produces correctly-formatted values because the error message tells the model exactly what format to use.

#### The extractor function

```python
# extractor/extractor.py
import instructor
from pydantic import BaseModel
from typing import TypeVar

T = TypeVar("T", bound=BaseModel)

_clients = {
    "openai":    instructor.from_provider("openai/gpt-4o-mini"),
    "anthropic": instructor.from_provider("anthropic/claude-sonnet-4-6"),
}

def extract(
    text: str,
    schema: type[T],
    provider: str = "openai",
    max_retries: int = 3,
) -> T:
    """Extract structured data from text into the given Pydantic schema."""
    client = _clients[provider]
    system = (
        "You are a careful data extractor. You never guess. "
        "If a field is not clearly present, return null for it. "
        "If the input is ambiguous, return the most conservative interpretation. "
        "You prefer missing data to wrong data."
    )
    return client.create(
        response_model=schema,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Extract the data from this text:\n\n{text}"},
        ],
        max_retries=max_retries,
    )
```

That's the entire core of the library. Three providers, one function, full type safety.

#### Batch mode

```python
# extractor/cli.py
import typer, jsonlines, json
from pathlib import Path
from extractor.extractor import extract
from extractor.schemas.receipt import Receipt
from extractor.schemas.job_listing import JobListing
from extractor.schemas.contact import Contact

SCHEMAS = {
    "receipt":      Receipt,
    "job_listing":  JobListing,
    "contact":      Contact,
}

app = typer.Typer()

@app.command()
def run(
    schema: str = typer.Argument(..., help="One of: receipt, job_listing, contact"),
    text: str = typer.Argument(None),
    file: Path = typer.Option(None, "--file", "-f"),
    provider: str = typer.Option("openai", "--provider"),
):
    if file:
        text = file.read_text()
    result = extract(text, SCHEMAS[schema], provider=provider)
    typer.echo(json.dumps(result.model_dump(), indent=2))

@app.command()
def batch(
    schema: str = typer.Argument(...),
    directory: Path = typer.Argument(...),
    output: Path = typer.Option("out.jsonl", "--output", "-o"),
    provider: str = typer.Option("openai"),
):
    success = 0
    failures = 0
    with jsonlines.open(output, "w") as writer:
        for txt_file in directory.glob("*.txt"):
            try:
                result = extract(txt_file.read_text(), SCHEMAS[schema], provider=provider)
                writer.write({"file": txt_file.name, "ok": True, "data": result.model_dump()})
                success += 1
            except Exception as e:
                writer.write({"file": txt_file.name, "ok": False, "error": str(e)})
                failures += 1
    typer.echo(f"Success: {success}, Failures: {failures}")
```

#### Eval set

For each schema, create a directory of **20+ real text examples** with **expected JSON outputs**. You can write them by hand, generate them with help from an LLM and then verify by hand, or find public datasets (receipts: various OCR benchmarks; job listings: scraped from job boards with permission).

```
eval_data/receipts/
├── clean_01.txt     ← a well-formatted receipt
├── clean_01.json    ← expected extraction
├── messy_01.txt     ← a partially-garbled OCR
├── messy_01.json
├── missing_items_01.txt  ← no line items
├── missing_items_01.json
...
```

Cover:
- Clean, well-formatted examples (should be nearly 100% correct)
- Messy, partially readable examples
- Examples missing key fields (to test the escape hatches)
- Non-English examples (even just one to see how the model handles)
- Edge cases (very long, very short, unusual formats)

#### Accuracy script

```python
# extractor/metrics.py
import json
from pathlib import Path
from extractor.extractor import extract

def field_accuracy(predicted: dict, expected: dict) -> dict[str, bool]:
    """Compare each field. Returns {field_name: is_correct}."""
    return {
        field: predicted.get(field) == expected.get(field)
        for field in expected.keys()
    }

def run_eval(schema_cls, eval_dir: Path, provider: str):
    total_fields = 0
    correct_fields = 0
    per_doc = []
    for txt in eval_dir.glob("*.txt"):
        expected = json.loads((txt.with_suffix(".json")).read_text())
        try:
            predicted = extract(txt.read_text(), schema_cls, provider=provider).model_dump()
            acc = field_accuracy(predicted, expected)
            total_fields += len(acc)
            correct_fields += sum(acc.values())
            per_doc.append({"file": txt.name, "accuracy": sum(acc.values()) / len(acc)})
        except Exception as e:
            per_doc.append({"file": txt.name, "error": str(e)})
    return {
        "field_accuracy": correct_fields / total_fields if total_fields else 0,
        "per_doc": per_doc,
    }
```

Run it:

```bash
playground eval receipt --provider openai
playground eval receipt --provider anthropic
```

Report field accuracy, per-document accuracy, and any hard-failure files. This is your evidence for which provider works best on your data.

### Stretch goals (pick ≥1)

- **Schema discovery.** Given a few example documents, ask the LLM to *propose* a Pydantic schema. Then use that schema for extraction. Bootstraps new schemas without hand-authoring.
- **Vision input.** Extend the extractor to accept image files (receipts, business cards). Uses Lesson 10's multimodal tips.
- **LLM-as-judge evaluation.** For fields with semantic variation (company names with slight formatting differences), use LLM-as-judge to decide if two extractions are equivalent.
- **Streaming partial extraction.** Use Instructor's `create_partial` to render the extracted object field-by-field in real time (nice for UIs).
- **Iterable extraction.** For documents containing multiple items (invoices with many line items, a catalog with many products), use `create_iterable` and compare the yielding-as-you-go experience.
- **Provider comparison dashboard.** A table showing each schema × each provider's accuracy and cost per document. Decide the best provider for each schema based on evidence.

---

## Evaluation rubric

- [ ] 3 Pydantic schemas defined, each with ≥1 validator and clear field descriptions
- [ ] Every optional field uses `X | None` with `Field(description=...)` explaining when to use None
- [ ] All closed enums use `Literal`
- [ ] Extraction function works against at least 2 providers interchangeably
- [ ] Batch mode processes a directory of files and emits JSONL
- [ ] Eval set of 20+ documents per schema, each with expected JSON output
- [ ] Eval script computes field-level accuracy per provider
- [ ] On clean examples, field accuracy ≥95%
- [ ] On messy examples, field accuracy ≥80%
- [ ] The model *never hallucinates* values on documents missing a field (verified manually on 5 examples)
- [ ] CLI installable via `pip install -e .`
- [ ] README documents the schemas, how to add new ones, and the eval results
- [ ] At least one stretch goal done
- [ ] You've used the tool on one real document of your own (a receipt, a listing, a business card) and captured the result

---

## Common pitfalls

- **Required fields without escape hatches.** The #1 cause of hallucination. If a field might legitimately be absent, make it `X | None`.
- **Not using `Literal` for enums.** You'll get variations like `"High"`, `"high"`, `"HIGH"`, `"High urgency"`. Use `Literal` to lock them down.
- **Validators without helpful error messages.** When a validator raises, Instructor sends the error back to the model. A message like `"Invalid"` tells the model nothing. A message like `"Phone must be E.164 format starting with +. Got: (555) 123-4567. Normalize by removing spaces and dashes."` gets the retry right almost every time.
- **Testing on only clean examples.** Clean examples hide the real problems. Make sure your eval set has at least 30% messy inputs.
- **Forgetting to set `max_retries`.** Default is 0 on some Instructor versions. Always set it explicitly. 3 is a good default.
- **Parsing the Instructor output as a dict instead of a Pydantic model.** You're losing type safety. Keep it as a model until serialization time.
- **Assuming schemas transfer between providers.** OpenAI's strict mode has stricter requirements than Anthropic's tool use. Test on both. Instructor mostly handles this, but not always.
- **Not handling the Anthropic "no strict mode" case.** Claude will occasionally produce a malformed output on the first try. Instructor's retries save you, but you need to budget for the extra cost.
- **Extracting fields you don't need.** Every field in your schema costs tokens and invites failure. Only model what your downstream code actually uses.
- **Not running the eval after schema changes.** Schemas are code. Every change gets an eval.

---

## Cost estimate

| Activity | Approximate cost |
|---|---|
| Schema iteration with cheap models | <$0.50 |
| Full eval on 60 docs × 2 providers | ~$1-2 |
| Buffer for stretch goals | ~$1 |
| **Total** | **~$2-3** |

Use Haiku 4.5 and GPT-4o-mini for development. Only run flagships for comparison benchmarks.

---

## What to deliver

```
structured-extractor/
├── pyproject.toml
├── extractor/
├── eval_data/
│   ├── receipts/
│   ├── job_listings/
│   └── contacts/
├── tests/
├── README.md            ← includes eval results
└── .env.example
```

README should include:
- The 3 schemas and what they extract
- Example invocations (CLI)
- Eval results: accuracy per schema × per provider, with a recommendation for which to use
- Known limitations (which messy inputs fail and why)
- How to add a new schema (step-by-step)

---

## Going further (after you finish)

- Add **10 more schemas** — medical records, legal contracts, scientific abstracts, whatever your day job touches. The pattern is the same; the schemas are the work.
- Build a **web scraper → extractor pipeline**. Scrape job listings from a public board, run them through the extractor, save to SQLite. Now you have a job search database.
- **Fine-tune a small open-weight model** on your eval data once you have enough examples (Module 1 Lesson 6). Compare cost and accuracy vs. Instructor + Haiku.
- **Turn it into a hosted API.** Wrap the extractor in FastAPI, deploy to a cheap VPS. This is how real SaaS LLM products work.

---

## References

- Instructor, *Documentation*. https://python.useinstructor.com/
- Pydantic, *Validators*. https://docs.pydantic.dev/latest/concepts/validators/
- OpenAI, *Structured outputs*. https://platform.openai.com/docs/guides/structured-outputs
- Anthropic, *Tool use with Claude*. https://docs.claude.com/en/docs/build-with-claude/tool-use
- Jason Liu, *Why I built Instructor*. https://jxnl.github.io/instructor/
- Eugene Yan, *Evals for LLM-based data extraction*. https://eugeneyan.com/writing/abstractive/

---

[← Previous Project](01-prompt-playground.md) | [Back to Prompt Engineering](../README.md) | [Next Project → Multi-Model Router](03-multi-model-router.md)
