# Project 1 — Prompt Playground

🟢 **Beginner** · ~4-6 hours · ~$2 in API credits

Every AI engineer ends up building something like this. A CLI that sends prompts to multiple providers, tweaks parameters, saves results, and lets you compare them side-by-side. By the end you'll have a tool you actually reach for every time you need to sanity-check a prompt — and you'll have built the mental model of "prompt + params → response → measurement" that the rest of this module is about.

---

## Prerequisites

- Finished **Lessons 1-3, 7, 12** (intro, messages, anatomy, role prompting, evals)
- Python 3.11+
- API keys for at least **3 providers** (see [Setup](#setup))
- Finished Module 1 Project 1 so you have the baseline for API calls

---

## What you'll build

A Python CLI — call it `playground` — that:

1. Sends a prompt to **3+ providers** (OpenAI, Anthropic, Groq by default; extensible to others) with a single command.
2. Lets you configure **temperature, top-p, max tokens, and system prompt** at the command line.
3. Displays responses **side-by-side** with per-model metadata: tokens, latency, cost.
4. **Streams** output when requested — prints tokens as they arrive, like ChatGPT's UI.
5. Saves every run (prompt, params, responses, timestamps, cost) to a **local SQLite** store.
6. Lets you **replay** any past run with different parameters to see how they change the output.
7. Loads prompts from **files** so multi-line prompts aren't hellish to type.
8. Supports simple **template substitution** — `{variable}` placeholders filled from CLI args.
9. Reads API keys from a **`.env` file**, never from command lines.

You should actively use this tool by the end of the project. If you wouldn't reach for it, it's not done.

---

## What you'll learn

- Multi-provider API abstraction (the same logical call against three different SDKs)
- Streaming responses and time-to-first-token
- Local persistence for experiment tracking
- Prompt templating with named variables
- Async execution to run providers in parallel
- Building a tool you'll actually use — not a throwaway script

---

## Tech stack

- **Python 3.11+**
- `openai`, `anthropic`, `groq` — provider SDKs
- `typer` — the CLI framework (cleaner than click for typed interfaces)
- `rich` — for side-by-side output and spinners
- `sqlite3` (stdlib) — for run storage
- `python-dotenv` — for API keys
- `pydantic` — for typed configs and run records
- `anyio` or `asyncio` — for parallel provider calls (stretch goal)

---

## Setup

```bash
mkdir prompt-playground && cd prompt-playground
python -m venv venv
source venv/bin/activate         # Unix
venv\Scripts\activate            # Windows

pip install openai anthropic groq typer rich pydantic python-dotenv
```

Project layout:

```
prompt-playground/
├── pyproject.toml
├── playground/
│   ├── __init__.py
│   ├── cli.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py         ← abstract Provider class
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── groq.py
│   ├── store.py            ← SQLite run log
│   ├── pricing.py           ← reused from Module 1 Project 2
│   └── models.py            ← Pydantic dataclasses
├── prompts/                 ← example prompts for the eval
│   ├── creative.md
│   ├── factual.md
│   └── reasoning.md
├── tests/
│   └── test_cli.py
├── .env.example
├── .gitignore
└── README.md
```

---

## Requirements

### Must have

#### Provider abstraction

Define a `Provider` protocol and implement it three times:

```python
# playground/providers/base.py
from typing import Protocol, Iterator
from playground.models import CompletionResult

class Provider(Protocol):
    name: str

    def complete(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> CompletionResult:
        ...

    def stream(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[str]:
        ...
```

Each provider implementation is ~30-50 lines. Normalize the response shape so downstream code doesn't care which provider produced it.

```python
# playground/models.py
from pydantic import BaseModel
from datetime import datetime

class CompletionResult(BaseModel):
    provider: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: str | None = None

class Run(BaseModel):
    id: str
    timestamp: datetime
    prompt: str
    system: str | None
    temperature: float
    top_p: float
    max_tokens: int
    results: list[CompletionResult]
```

#### CLI commands

Minimum:

```bash
# Run a prompt against all providers
playground run "Write a haiku about bottles" \
    --temperature 0.7 \
    --system "You are a terse haiku master"

# Same, but load the prompt from a file
playground run --file prompts/creative.md

# Stream output from one provider (the fastest way to "feel" a prompt)
playground stream "Write a haiku about bottles" --provider anthropic

# Substitute variables in a template prompt
playground run --file prompts/review.md \
    --var product="Falo Bottles Summit" \
    --var tone="warm"

# List past runs
playground history --limit 10

# Replay a past run with different params
playground replay <run_id> --temperature 1.0

# Show cumulative spend for the current month
playground spend
```

Use Typer's command decorator for each command.

#### Streaming side-by-side

When run without `--provider`, `run` should call all providers **concurrently** and display them side-by-side in a Rich `Table` or `Columns` layout. For streaming, use Rich's `Live` context to update the columns as tokens arrive.

```python
from rich.live import Live
from rich.table import Table

async def run_all(prompt, system, providers):
    tasks = [p.stream(prompt, system, 0.7, 1.0, 1024) for p in providers]
    # use asyncio.gather or anyio.create_task_group
    ...
```

If async feels too ambitious for v1, ship with sequential calls and add parallelism in v2. The experience improvement from parallel calls is huge, though — prioritize it as soon as possible.

#### Run storage (SQLite)

Every run gets written to `~/.prompt-playground/runs.db` (or a local `runs.db`, your choice). Schema:

```sql
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    prompt TEXT NOT NULL,
    system TEXT,
    temperature REAL NOT NULL,
    top_p REAL NOT NULL,
    max_tokens INTEGER NOT NULL,
    total_cost_usd REAL NOT NULL
);

CREATE TABLE results (
    run_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    finish_reason TEXT,
    PRIMARY KEY (run_id, provider),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
```

Keep it simple. No ORM — stdlib `sqlite3` is perfect.

#### Template substitution

```
# prompts/review.md
You are reviewing a product for Falo Bottles.

Product: {product}
Tone: {tone}

Write a 3-sentence review that matches the tone.
```

```bash
playground run --file prompts/review.md \
    --var product="Summit 20oz" \
    --var tone="enthusiastic"
```

The CLI reads the file, substitutes `{product}` and `{tone}`, and sends the result. Either use Python's `str.format()` or Jinja2 (overkill for v1).

#### Config via .env

```
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

Fail fast at startup with a clear error if a key is missing. Never log the key.

### Stretch goals (pick ≥1)

- **Diff mode.** After running a prompt on all providers, highlight where responses disagree. Use `difflib` or a simple word-level diff.
- **Gradio UI.** Wrap the same functions in a Gradio web app for non-CLI use. ~30 lines.
- **Prompt library.** A `~/.prompt-playground/library/` directory of saved prompts with metadata, tags, and a `playground library search` command.
- **Ollama provider.** Add a local provider so you can compare cloud models against a 7B model running on your machine (Module 1 Lesson 12).
- **Eval mode.** Given a `prompts/*.md` directory and an expected-output column, score each provider against the expectations. This is a preview of Project 2 from Module 5 (Observability).
- **Prompt variation runs.** `playground run prompt.md --sweep temperature=0.1,0.5,0.9` runs the same prompt at each setting and shows a comparison table.
- **Cost alert.** Show a warning when a single run is projected to cost more than $X.

---

## Starter scaffold

```python
# playground/providers/anthropic.py
import time
from anthropic import Anthropic
from playground.models import CompletionResult
from playground.pricing import cost_usd

_client = Anthropic()

class AnthropicProvider:
    name = "anthropic"
    model = "claude-haiku-4-5"

    def complete(self, prompt, system, temperature, top_p, max_tokens):
        start = time.perf_counter()
        resp = _client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        text = "".join(b.text for b in resp.content if b.type == "text")
        return CompletionResult(
            provider=self.name,
            model=self.model,
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=elapsed_ms,
            cost_usd=cost_usd(self.model, resp.usage.input_tokens, resp.usage.output_tokens),
            finish_reason=resp.stop_reason,
        )

    def stream(self, prompt, system, temperature, top_p, max_tokens):
        with _client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
        ) as s:
            for text in s.text_stream:
                yield text
```

```python
# playground/cli.py
import typer
from rich.console import Console
from rich.table import Table
from playground.providers.openai import OpenAIProvider
from playground.providers.anthropic import AnthropicProvider
from playground.providers.groq import GroqProvider
from playground.store import save_run
import uuid, datetime

app = typer.Typer()
console = Console()

PROVIDERS = [OpenAIProvider(), AnthropicProvider(), GroqProvider()]

@app.command()
def run(
    prompt: str = typer.Argument(None),
    file: str = typer.Option(None, "--file", "-f"),
    system: str = typer.Option(None, "--system", "-s"),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
    top_p: float = typer.Option(1.0, "--top-p"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
):
    if file:
        prompt = open(file).read()
    if not prompt:
        typer.echo("No prompt provided.", err=True)
        raise typer.Exit(1)

    results = []
    for provider in PROVIDERS:
        console.print(f"[dim]Calling {provider.name}...[/dim]")
        result = provider.complete(prompt, system, temperature, top_p, max_tokens)
        results.append(result)

    # Display side-by-side
    table = Table(title=f"Results (temp={temperature}, top_p={top_p})")
    for r in results:
        table.add_column(f"{r.provider} ({r.model})")
    table.add_row(*[r.text for r in results])
    console.print(table)

    # Metadata
    meta = Table(title="Metadata", show_header=True)
    meta.add_column("Provider")
    meta.add_column("Tokens in")
    meta.add_column("Tokens out")
    meta.add_column("Latency (ms)")
    meta.add_column("Cost (USD)")
    for r in results:
        meta.add_row(r.provider, str(r.input_tokens), str(r.output_tokens),
                     f"{r.latency_ms:.0f}", f"${r.cost_usd:.5f}")
    console.print(meta)

    # Save the run
    save_run(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.datetime.utcnow(),
        prompt=prompt,
        system=system,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        results=results,
    )

if __name__ == "__main__":
    app()
```

```toml
# pyproject.toml
[project]
name = "prompt-playground"
version = "0.1.0"
dependencies = [
    "openai", "anthropic", "groq",
    "typer", "rich", "pydantic", "python-dotenv"
]

[project.scripts]
playground = "playground.cli:app"
```

```bash
pip install -e .
playground run "Say hi"
```

---

## Evaluation rubric

- [ ] CLI installs via `pip install -e .` and `playground` is on your PATH
- [ ] `playground run` calls all 3 providers and displays results side-by-side
- [ ] `playground stream` streams from a single provider with tokens appearing in real time
- [ ] Temperature, top-p, max-tokens, system prompt all configurable via CLI flags
- [ ] Prompts can be loaded from a file with `--file`
- [ ] Template variables work: `--var key=value` substitutes `{key}` in the prompt
- [ ] Every run is saved to SQLite with timestamp, params, all provider responses, and cost
- [ ] `playground history` shows past runs; `playground replay <id>` re-executes a run with optional param overrides
- [ ] `playground spend` sums the cumulative cost from the SQLite log
- [ ] `.env` loading works; missing keys produce a clear startup error
- [ ] Pricing config is loaded from a dict (reuse Module 1 Project 2's pattern)
- [ ] At least one stretch goal completed
- [ ] README with 5+ example invocations and screenshots of Rich output
- [ ] You've used the tool on at least 3 real prompts from your own work and documented observations

---

## Common pitfalls

- **Hardcoding model names.** Put them in a config so you can swap "cheap default" vs "flagship" without code changes.
- **Sequential calls for parallel work.** Three providers sequentially is ~6 seconds of waiting. Parallel is ~2 seconds. Worth the async complexity.
- **Streaming from multiple providers with Rich's `Live`.** This is tricky — you need to use `Live.update(Table(...))` inside an async for-loop. If you get stuck, ship with non-streaming multi-provider and streaming single-provider first. Don't let this block the whole project.
- **Not handling rate limits.** Free-tier accounts (especially Groq) hit rate limits fast. Add a `try/except` that prints a clear message and continues with the remaining providers.
- **Running tests against production keys.** Use Groq's free tier for iteration. Save paid API calls for the final demo runs.
- **Storing the full prompt and response in SQLite as strings.** That works fine until you have a 50k-token prompt. SQLite has a 1GB TEXT column limit, so it's really fine for any prompt you'd call from a CLI.
- **Logging API keys.** Never print `os.environ` or any config containing keys. Triple-check your logging.
- **Over-designing the provider abstraction.** A Protocol with two methods is enough. Don't build an inheritance tree.
- **Forgetting `max_tokens` on Anthropic.** Anthropic requires it. OpenAI doesn't. If you copy-paste OpenAI code into Anthropic code, you'll get an error.
- **Not making it installable.** `python playground/cli.py` works for testing. `playground run ...` is what you'll actually use. Install it early.

---

## Cost estimate

Realistic budget for this project:

| Activity | Cost |
|---|---|
| Iteration runs (with cheap models) | <$0.50 |
| Demo / stretch-goal runs with flagship models | $1-2 |
| Unexpected screw-ups | $1-2 |
| **Total** | **~$2-5** |

Use Groq's free tier + Haiku + GPT-5-nano for iteration. Run flagships only for final tests.

---

## What to deliver

```
prompt-playground/
├── pyproject.toml
├── playground/              ← source
├── prompts/                 ← example prompts
├── tests/
├── .env.example             ← template, no real keys
├── .gitignore
├── runs.db                  ← optionally include a few real runs
└── README.md
```

README should include:
- One-line description
- Install and setup
- 5+ example invocations with rendered output (screenshots or text)
- Which 3 providers are supported and how to add more
- Known limitations
- 2-3 observations from using it on real prompts

---

## Going further (after you finish)

- Add a **provider for your local Ollama** (Module 1 Lesson 12). Compare a local Qwen or Llama model against the cloud ones.
- Build an **eval mode** that takes `prompts/*.md` with expected-output front-matter and produces a pass/fail table — groundwork for Project 3.
- Publish a **small demo video** (30 seconds, terminal recording) and drop it in the README. Recruiters love these.
- **Ship to PyPI** with a proper package name. Your first published tool.

---

## References

- Typer, *Modern CLI framework for Python*. https://typer.tiangolo.com/
- Rich, *Terminal formatting and live displays*. https://rich.readthedocs.io/
- Anthropic, *Messages API streaming*. https://docs.claude.com/en/api/messages-streaming
- OpenAI, *Chat completions streaming*. https://platform.openai.com/docs/api-reference/streaming
- Groq, *API documentation*. https://console.groq.com/docs/quickstart
- SQLite3 stdlib docs. https://docs.python.org/3/library/sqlite3.html

---

[← Back to Prompt Engineering](../README.md) | [Next Project → Structured Data Extractor](02-structured-extractor.md)
