# Lesson 8 — Structured Outputs in Practice

> **The single sentence version:** Module 1 taught you that strict-mode structured outputs guarantee schema-valid JSON — this chapter teaches you how to design the schemas, write the prompts, and handle the edge cases so your production pipelines actually work.

You already know the mechanics of structured outputs from Module 1 Lesson 10: strict mode, JSON schemas, tool use, Pydantic models. This chapter is about the *craft* of using them well. What makes a good schema? When do you use XML tags instead of JSON? How do you handle partial extractions, validation failures, and schema evolution? These are the questions you'll live with once you start building real pipelines.

---

## The mental model

Before we get into tactics, one mental model to hold: **structured output design is API design**. Your schema is the contract between the LLM and your downstream code. Like any API, it needs to be:

- **Clear**: each field's purpose is obvious from its name and description
- **Minimal**: every field earns its place; no speculative fields
- **Stable**: you can version it without breaking callers
- **Validated**: inputs that don't match the schema are rejected, not silently accepted

Everything else in this chapter is just "how to honor those four properties when the upstream producer is an LLM instead of a human developer."

---

## Pydantic: the Python workhorse

The cleanest way to define schemas in Python (for either OpenAI's structured outputs or Anthropic's tool use) is **Pydantic**. A Pydantic model is both a type hint and a schema. The SDKs convert it to JSON schema for you and parse responses back into typed objects.

### The basic pattern

```python
from pydantic import BaseModel, Field
from openai import OpenAI

class ContactInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    email: str | None = Field(description="Email address, or None if not present")
    phone: str | None = Field(description="Phone number, normalized to +CC format")
    company: str | None = Field(description="Company or organization, if mentioned")

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract contact info from the text."},
        {"role": "user",   "content": "Call me at Acme — Jane Doe, jane@acme.co, +65 9123 4567."},
    ],
    response_format=ContactInfo,
)

contact: ContactInfo = response.choices[0].message.parsed
print(contact.name)     # "Jane Doe"
print(contact.email)    # "jane@acme.co"
```

You get a typed Python object, not a string you have to parse. If the model's output doesn't match the schema, the SDK raises an error before you see it. This is the quality-of-life baseline for 2026 Python.

### Field descriptions are instructions

A key technique: **use `Field(description=...)` to give each field a tiny instruction.** The model reads these descriptions and uses them to decide what to put in each field.

```python
class Review(BaseModel):
    sentiment: Literal["positive", "negative", "mixed", "unknown"] = Field(
        description="Overall sentiment. Use 'unknown' if the review is too neutral or "
                    "factual to classify. Do not force a classification on unclear cases."
    )
    key_complaint: str | None = Field(
        description="The single most important complaint the reviewer makes, "
                    "paraphrased in ≤10 words. None if the review has no complaints."
    )
    would_recommend: bool | None = Field(
        description="Whether the reviewer explicitly recommends or advises against. "
                    "None if the review doesn't take a clear stance."
    )
```

Notice how each description is a micro-prompt. It defines the semantics, provides an escape hatch for ambiguous cases, and specifies constraints (length, format) inline. The model treats these descriptions as part of the instruction — use them generously.

### Literal types for enums

For closed-world categorical fields, use `Literal`:

```python
from typing import Literal

class Classification(BaseModel):
    category: Literal["bug", "feature", "question", "spam"]
```

The JSON schema generated from this will have an `enum` constraint, and strict mode will *literally prevent* the model from producing any value outside that list. You can't get back `"Bug"` or `"bugs"` or `"defect"` — only exactly `"bug"`. This is much more reliable than trying to enforce enumerated values in the prompt.

### Optional fields

Fields that might legitimately be absent should be marked optional using a union with `None`:

```python
class Book(BaseModel):
    title: str
    author: str
    isbn: str | None = None        # may not be present
    publication_year: int | None = None
```

**Important caveat for OpenAI strict mode**: strict mode requires *every field* to be listed in `required`, but it allows `None` as a value for union types. So you don't write `Optional[str]` and leave it out; you write `str | None` and make it required-with-nullable. This is a subtle point that will cost you an afternoon the first time you hit it.

### Validators

Pydantic validators let you enforce rules that JSON schema can't express:

```python
from pydantic import BaseModel, Field, field_validator

class Event(BaseModel):
    title: str = Field(min_length=5, max_length=100)
    start_time: str = Field(description="ISO 8601 datetime")
    duration_minutes: int = Field(ge=5, le=480)

    @field_validator("start_time")
    @classmethod
    def validate_iso8601(cls, v):
        from datetime import datetime
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("start_time must be ISO 8601 format")
        return v
```

Validators run after the LLM responds. If a validator fails, the SDK (and the Instructor library — more below) can automatically retry with the error as feedback.

---

## Instructor: retries, partials, and the ergonomics layer

**Instructor** is a thin wrapper around the OpenAI / Anthropic / Gemini SDKs that adds features the base SDKs lack:

- **Automatic retries on validation failure** — if the model produces something your Pydantic model rejects, Instructor sends the validation error back to the model as a hint and retries up to `max_retries` times.
- **Streaming partial objects** — watch the response build up field by field in real time (useful for UIs).
- **Iterable extraction** — extract multiple objects from one input in a streaming fashion.
- **15+ provider support** — same API for OpenAI, Anthropic, Gemini, Ollama, DeepSeek, Mistral, and more.
- **LLM-based validators** — validate outputs using another LLM call ("is this appropriate?").

A typical Instructor call looks like:

```python
import instructor
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    occupation: str

client = instructor.from_provider("openai/gpt-4o-mini")

profile = client.create(
    response_model=UserProfile,
    messages=[{"role": "user", "content": "John is a 30-year-old software engineer."}],
    max_retries=3,
)
# profile is already a typed UserProfile object
```

The `max_retries=3` is the killer feature. If the model fails validation on the first try — maybe it returned the age as a string, or omitted a field — Instructor sends a message like "The previous response failed validation: age must be an integer" and asks the model to try again. Usually one retry is enough.

This is especially valuable on Anthropic, which (as of early 2026) doesn't have a strict-mode equivalent to OpenAI's `strict: true`. Instructor's retry loop effectively simulates it: you *will* get a valid object by the end, even if it takes 2-3 attempts.

### Streaming partials

For UIs that want to show structured data as it builds up:

```python
for partial in client.create_partial(
    response_model=ArticleSummary,
    messages=[{"role": "user", "content": text}],
):
    render(partial)   # shows the object as each field gets filled in
```

Each iteration of the loop gives you a progressively more-complete object. Fields appear as the model generates them. The UX is like a progress bar, but for structured data.

### Extracting lists

```python
class Product(BaseModel):
    name: str
    price: float

for product in client.create_iterable(
    response_model=Product,
    messages=[{"role": "user", "content": "Parse this catalog: ..."}],
):
    save_to_db(product)   # each product yields as it's extracted
```

For inputs that contain many items (a catalog, a list of events, a table of records), this lets you process them one at a time instead of waiting for the whole list to complete.

---

## JSON schema vs. XML tags: when to use which

You have two main ways to get structured output from an LLM: **JSON schemas** (OpenAI strict mode, Anthropic tool-use) and **XML tags** (the Claude-specific convention from Anthropic's docs).

They're complementary. Here's when to use each.

### JSON schemas — use when

- The output is **machine-consumed** by downstream code that expects JSON
- You need **strict shape guarantees** that the code can rely on
- The output has **many fields** or **nested structure**
- You're using **OpenAI** (strict mode) or want maximum tool-like reliability on any provider

Pydantic + Instructor + strict mode is the production default in Python for any data-extraction pipeline.

### XML tags — use when

- The output is **human-readable prose with structured markers**, not pure data
- You want to **separate reasoning from the answer** (Lesson 5)
- The output mixes **free-form text** with tagged regions (think: essays with callouts)
- You're on **Claude** and want the model's best-case performance on complex prompts

From Anthropic's own docs: XML tags help Claude parse complex prompts unambiguously, especially when prompts mix instructions, context, examples, and variable inputs. Claude is particularly good at producing XML-tagged output because it was trained on lots of examples structured that way.

An XML-tag example — summarize a document with both a summary and a list of risks:

```
Summarize the document below. Put a one-sentence summary inside <summary> tags
and a list of key risks inside <risks> tags (one <risk> tag per risk).

<document>
{text}
</document>
```

Output:
```xml
<summary>
Global steel prices surged 18% in Q3 as demand from Indian infrastructure projects
outpaced supply recovery.
</summary>
<risks>
  <risk>Supply chain disruptions from the Red Sea route affecting timeline</risk>
  <risk>Currency volatility in the Indian rupee</risk>
  <risk>Competition from Chinese oversupply that could depress prices in Q4</risk>
</risks>
```

Your code can extract this with a simple regex or HTML/XML parser. It's not as rigidly validated as JSON schema, but it's simpler to write and the output is often more readable when you're debugging.

### The hybrid approach

You can mix them: use XML tags in the *prompt* to structure your instructions, and ask for JSON in the *output* for downstream consumption.

```
Extract product information from the text.

<examples>
  <example>
    <text>MSRP $24.99. Falo Bottle Summit 20oz in matte black.</text>
    <output>{"name": "Falo Bottle Summit 20oz", "color": "matte black", "price_usd": 24.99}</output>
  </example>
</examples>

<text_to_process>
{text}
</text_to_process>

Respond with only JSON matching the schema.
```

The XML structures the *prompt* (instructions, examples, input) so the model can tell the parts apart. The *response* is JSON for your code. Best of both worlds.

---

## Designing schemas that work

A few principles for schemas the model can reliably fill in:

### 1. One thing per field

Don't design fields that require the model to make multiple independent decisions simultaneously.

**Bad:**
```python
class Ticket(BaseModel):
    category_and_urgency: str   # "bug-high", "feature-low", etc.
```

**Good:**
```python
class Ticket(BaseModel):
    category: Literal["bug", "feature", "question", "spam"]
    urgency: Literal["low", "medium", "high", "critical"]
```

Two separate decisions, each enforced independently. Cleaner for both the model and your code.

### 2. Include escape hatches

Always give the model a way to say "I don't know" or "not applicable." Otherwise it will hallucinate to fill required fields.

```python
class ExtractedPrice(BaseModel):
    amount: float | None = Field(
        description="The price in dollars, or None if no price is explicitly stated"
    )
    currency: Literal["USD", "EUR", "SGD", "unknown"] = Field(
        description="The currency. Use 'unknown' if not explicitly specified."
    )
```

Without the `None` and `"unknown"`, the model has to make something up. With them, you get missing data instead of wrong data, which is usually what you want.

### 3. Prefer enums over free strings

Whenever a field has a known set of valid values, use `Literal`. Free strings let the model get creative in ways that break your downstream code.

### 4. Field order matters

The model generates fields in the order they appear in the schema. Put *simpler* fields before *harder* ones, so the easy decisions set context for the hard ones.

```python
# Bad ordering: the model has to decide the final answer before it's seen any reasoning
class Analysis(BaseModel):
    final_decision: Literal["approve", "reject"]
    reasoning: str
    risks: list[str]

# Good ordering: reasoning and risks first, final decision last
class Analysis(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning about the request")
    risks: list[str] = Field(description="List of risks identified during reasoning")
    final_decision: Literal["approve", "reject"] = Field(
        description="Based on the reasoning and risks above"
    )
```

This is essentially built-in chain-of-thought (Lesson 5). The model reasons into the early fields and uses that reasoning when filling in the later ones.

### 5. Don't over-nest

Deeply nested schemas are hard for the model to get right consistently. Flatten when you can.

**Hard to fill:**
```python
class Address(BaseModel):
    street: Street
    city: City
    country: Country

class Street(BaseModel):
    number: int
    name: str
    type: Literal["street", "avenue", "road", "lane"]
```

**Easier:**
```python
class Address(BaseModel):
    street_number: int
    street_name: str
    street_type: Literal["street", "avenue", "road", "lane"]
    city: str
    country: str
```

Same information, half the schema depth, dramatically higher extraction success rate.

### 6. Add length / range bounds

```python
summary: str = Field(min_length=20, max_length=200)
confidence: float = Field(ge=0.0, le=1.0)
tags: list[str] = Field(min_length=3, max_length=7)
```

These show up in the JSON schema and are enforced by Pydantic on parse. The model sees them and tries to comply; Pydantic then verifies. Much more reliable than hoping the model counts right.

---

## The retry-with-validation loop

Even with strict mode, you'll sometimes get outputs that technically validate against the schema but are semantically wrong — an empty list, a default value, a field that's present but meaningless. For these cases, you need validators + retry.

The pattern, pulled together:

```python
import instructor
from pydantic import BaseModel, field_validator, ValidationError

class Summary(BaseModel):
    headline: str
    bullet_points: list[str]

    @field_validator("headline")
    @classmethod
    def headline_not_generic(cls, v):
        if v.lower().startswith(("article about", "summary of", "overview of")):
            raise ValueError(
                "The headline must not be generic. Write a specific, concrete headline "
                "that captures the article's main point."
            )
        return v

    @field_validator("bullet_points")
    @classmethod
    def bullets_substantive(cls, v):
        if len(v) < 3:
            raise ValueError("Must have at least 3 bullet points")
        for b in v:
            if len(b.split()) < 5:
                raise ValueError(
                    f"Bullet point '{b}' is too short. Each bullet must be a "
                    f"substantive sentence, at least 5 words."
                )
        return v

client = instructor.from_provider("anthropic/claude-sonnet-4-6")

summary = client.create(
    response_model=Summary,
    messages=[{"role": "user", "content": "Summarize this article: ..."}],
    max_retries=3,
)
```

The first attempt might produce generic headlines or short bullets. The validator rejects them, Instructor sends the error back to the model ("Must have at least 3 bullet points"), and the model tries again. Usually one retry is enough. This is the core pattern for "semantic validation, not just structural."

---

## Partial extractions: when the model can only do part of the job

Not every document contains every field. Your schema should expect this.

```python
class JobPosting(BaseModel):
    title: str                         # required — every job has a title
    company: str | None = None         # usually present
    location: str | None = None        # often missing
    salary_min: int | None = None      # rarely explicit
    salary_max: int | None = None      # rarely explicit
    seniority: Literal["junior", "mid", "senior", "staff", "unknown"] = "unknown"
    remote: Literal["yes", "no", "hybrid", "unknown"] = "unknown"
```

Only `title` is strictly required (marked with `str`, no union). Everything else defaults to `None` or `"unknown"`. The downstream code knows that "None" means "field was not present in the input" and handles it accordingly.

This is much cleaner than making everything required and getting hallucinated values, or making everything optional without defaults and having to check each field for presence.

---

## Schema evolution

Schemas change over time. You add fields, rename fields, widen enums. How do you do this without breaking running systems?

**Rules:**

- **Adding a new optional field is safe.** Old prompts/old models will just leave it `None`; new code that uses it still works.
- **Adding a new required field is a breaking change.** Either add it as optional first (with a migration plan), or version your schema (`UserV2` alongside `UserV1`).
- **Renaming a field is a breaking change.** Keep the old name as a deprecated alias if you can; otherwise version.
- **Widening an enum is safe.** Adding `"urgent"` to an existing `Literal["low", "medium", "high"]` → `Literal["low", "medium", "high", "urgent"]` works; old outputs are still valid.
- **Narrowing an enum is a breaking change.** Any existing output using the removed value will fail validation.

In practice: version your schemas the same way you'd version an API. `UserV1`, `UserV2`, with a migration layer. Tag the version in your telemetry so you can tell which version produced which rows. Don't try to be clever — schema versioning is boring for a reason.

---

## Common pitfalls

- **Designing schemas without thinking about downstream code.** Every field should correspond to something the consuming code actually needs. Speculative fields cost tokens and confuse the model.
- **Not using `Literal` for closed sets.** If there are 5 valid values, enforce them. Don't rely on the prompt to produce only those values.
- **Required fields without escape hatches.** Force-fitting answers produces hallucinations. Let the model say "unknown."
- **Nesting 3+ levels deep.** Flatten when you can. Deep nesting is where extraction errors hide.
- **Not writing validators.** Structural validation isn't enough — if you care whether the headline is generic, write a validator that checks for generic headlines.
- **Forgetting that Anthropic doesn't have strict mode.** Without Instructor's retries, Claude will occasionally produce a field with the wrong type. Budget for retries.
- **Putting validator logic in downstream code instead of the schema.** If the validation rule belongs to the schema, put it in the schema. Downstream code is the wrong place to discover that the LLM produced a generic headline.
- **Not testing with real data.** Schemas that look great on paper sometimes fail on messy real inputs. Always validate on 20+ real examples before shipping.
- **Using JSON when XML tags would be clearer.** Don't cram essay-shaped content into JSON just because JSON is "more structured." Match the format to the content.

---

## What to remember from this lesson

- Schema design is API design: clear, minimal, stable, validated.
- Pydantic models + `Field(description=...)` give you typed Python and micro-instructions in one place.
- Use `Literal` for enums. Use `X | None` for optional fields. Use validators for semantic constraints.
- Instructor adds retries, streaming partials, and provider-agnostic APIs. On Anthropic (no strict mode), Instructor's retries are essential.
- JSON schemas for machine consumption; XML tags for human-readable structured prose. Mix them in one prompt when helpful.
- Field order matters — put reasoning fields before decision fields so the model has context when making the decision.
- Give the model escape hatches ("unknown," `None`) so it returns missing data instead of hallucinations.
- Version your schemas like APIs. Adding optional fields is safe; everything else is not.
- Always test schemas on real, messy data before shipping.

Next: prompt templates, caching patterns, and the basics of making reusable prompts at scale.

---

## References

- Pydantic, *Documentation*. https://docs.pydantic.dev/latest/
- Instructor, *Documentation*. https://python.useinstructor.com/
- OpenAI, *Structured outputs guide*. https://platform.openai.com/docs/guides/structured-outputs
- OpenAI Cookbook, *Structured Outputs intro*. https://cookbook.openai.com/examples/structured_outputs_intro
- Anthropic, *Tool use with Claude*. https://docs.claude.com/en/docs/build-with-claude/tool-use
- Anthropic, *XML tags in prompts*. https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices
- Jason Liu, *Why Instructor exists*. https://jxnl.github.io/instructor/why/

---

[← Lesson 7](07-role-prompting.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 9: Templates and Caching](09-templates-and-caching.md)
