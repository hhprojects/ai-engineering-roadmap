# Project 2 — Token Economics Calculator

🟡 **Intermediate** · ~5-7 hours · <$1 in API credits

Tokens are the currency of LLMs, and most engineers fly blind on them. Build a tool that makes token counts and costs visible across providers — a utility you'll genuinely reach for every time you're sizing a prompt or estimating a monthly bill. This project is where "I know roughly what tokens are" turns into "I can argue about cost trade-offs with numbers."

---

## Prerequisites

- Finished **Lessons 2, 8, 11** (tokenization, model families, token economics)
- Finished **Project 1** (so you already have API keys and pricing context)
- Python 3.11+
- Comfortable with CLI tools and basic config files

---

## What you'll build

A CLI tool (optional web UI stretch) that:

1. Accepts text input from **stdin**, a file, or a CLI argument.
2. **Tokenizes** the input with at least two different tokenizers — one for OpenAI models, one for a Claude or Llama model — and shows the token counts side-by-side.
3. Estimates **cost across ≥5 models** (mix of frontier and cheap, across ≥2 providers).
4. **Ranks** the models cheapest to most expensive for a typical completion of the given prompt.
5. Supports a **`--budget $X`** flag that auto-selects the best model under a given cost per request.
6. Handles **batch mode** — point it at a file of prompts, get aggregate cost estimates.
7. Loads pricing from a **config file**, not hardcoded.
8. Handles edge cases: empty input, very long input, non-English text, code, JSON.

You should be proud of this tool. Polish it. The goal is that six months from now you're still reaching for it.

---

## What you'll learn

- How different tokenizers produce different counts for the same text (BPE variations, SentencePiece, byte-level differences)
- Token count asymmetry across English vs. code vs. multilingual text
- Building practical Python CLIs with `click` or `typer`
- Cost modeling for LLM workloads, including input/output asymmetry
- Loading configuration from files instead of hardcoding
- Writing a tool that other engineers could use, not just a script for yourself

---

## Tech stack

- **Python 3.11+**
- `tiktoken` — OpenAI's tokenizer library (fast, accurate for GPT models)
- `anthropic` — Anthropic SDK (has a token counting endpoint for Claude)
- `click` or `typer` — CLI framework (typer is nicer; click is more common)
- `rich` — optional but highly recommended for colorful output
- `pyyaml` or `tomli` — for loading the pricing config
- `pydantic` — optional, for validating the config file

For the stretch "web UI" goal:
- `gradio` — turn a Python function into a web form in ~10 lines

---

## Setup

```bash
mkdir token-economics && cd token-economics
python -m venv venv
source venv/bin/activate   # Unix
venv\Scripts\activate      # Windows

pip install tiktoken anthropic typer rich pyyaml pydantic
```

Then structure the project:

```
token-economics/
├── pyproject.toml       ← make it installable as a CLI tool
├── token_economics/
│   ├── __init__.py
│   ├── cli.py           ← main entry point
│   ├── tokenizers.py    ← wrapper around each tokenizer
│   ├── pricing.py       ← loads and queries the pricing config
│   └── models.py        ← dataclasses
├── pricing.yaml         ← the pricing config
├── tests/
│   └── test_tokenizers.py
└── README.md
```

---

## Requirements

### Must have

#### Tokenization

- Use `tiktoken` to tokenize for **GPT models**. The `cl100k_base` encoding covers GPT-4, GPT-4o, GPT-5. The `o200k_base` encoding covers newer models — check `tiktoken.encoding_for_model("gpt-5.4")` to get the right one.
- Use the **Anthropic API's token counting endpoint** to tokenize for **Claude** (the `messages.count_tokens` method does this without burning output tokens).
- Optionally, use the `transformers` library (or a pre-downloaded `.json` tokenizer) to tokenize for **Llama** or **Qwen** to illustrate how much open-weight tokenizers differ.

Expected output for a single input:

```
$ echo "Bonjour, le monde!" | token-econ count

Tokens per model:
  GPT-4o (tiktoken cl100k_base)        6 tokens
  GPT-5.4 (tiktoken o200k_base)        5 tokens
  Claude Sonnet 4.6 (Anthropic)        6 tokens
  Llama 3.1 (SentencePiece)            8 tokens
```

#### Cost estimation

For each model in your pricing config, compute:

```
estimated_cost = input_tokens × input_price_per_mtok / 1M
               + estimated_output_tokens × output_price_per_mtok / 1M
```

You need a way to estimate output tokens. Three options, in order of cleverness:

1. **Flat assumption** — assume output = X × input, where X defaults to 0.5 and is configurable (`--output-ratio 0.5`).
2. **Category-aware** — let the user pass `--task qa|summary|code|creative` and map each to a typical output length.
3. **Actually generate a sample** — call one cheap model, see how long it responds, extrapolate. Overkill but impressive.

Flat assumption is fine for this project. Add it to the CLI as a flag.

#### CLI interface

Minimum commands:

```bash
token-econ count [TEXT] [--file PATH] [--stdin]
    # Show token counts across providers

token-econ cost [TEXT] [--output-ratio 0.5] [--budget 0.01]
    # Show estimated costs across models, optionally filtered by budget

token-econ batch PATH [--output-ratio 0.5]
    # Process a file of prompts (one per line or JSON array), show totals

token-econ models
    # List all models in the config with their prices
```

#### Pricing config

Keep prices in `pricing.yaml` (or `pricing.toml`), **not** hardcoded:

```yaml
models:
  - name: claude-opus-4-6
    provider: anthropic
    input_per_mtok: 5.00
    output_per_mtok: 25.00
    context_window: 1_000_000
    tokenizer: anthropic

  - name: claude-sonnet-4-6
    provider: anthropic
    input_per_mtok: 3.00
    output_per_mtok: 15.00
    context_window: 1_000_000
    tokenizer: anthropic

  - name: gpt-5.4
    provider: openai
    input_per_mtok: 5.00
    output_per_mtok: 20.00
    context_window: 400_000
    tokenizer: tiktoken_o200k_base

  - name: gpt-5.4-mini
    provider: openai
    input_per_mtok: 0.50
    output_per_mtok: 2.00
    context_window: 400_000
    tokenizer: tiktoken_o200k_base

  - name: gemini-3-flash
    provider: google
    input_per_mtok: 0.15
    output_per_mtok: 0.60
    context_window: 1_000_000
    tokenizer: tiktoken_cl100k_base  # approximation

  - name: deepseek-v3-2
    provider: deepseek
    input_per_mtok: 0.30
    output_per_mtok: 1.20
    context_window: 128_000
    tokenizer: tiktoken_cl100k_base  # approximation
```

Validate it with Pydantic on load. Fail loudly if fields are missing.

#### Edge cases you must handle

- Empty input → clear error message, non-zero exit code
- Input larger than a model's context window → show a warning next to that model in the output
- Non-English text (test with at least one Chinese, one Arabic, one emoji-heavy prompt)
- Code (test with a block of Python)
- JSON (test with a deeply nested document)
- Very long input (10k+ tokens)

Don't silently succeed on edge cases. Either handle them or raise with a clear message.

### Stretch goals (pick ≥1)

- **Token visualizer.** For a given model, print the input with token boundaries color-coded — makes the tokenizer behavior visible.
- **Context utilization.** Show "this prompt is 12% of gpt-5.4's 400k window" next to each model.
- **Cumulative spending ledger.** A local SQLite file that tracks how much you'd have spent if you'd made these calls. Running totals per model, per day.
- **Web UI.** Wrap the core functions with Gradio for a pastable interface. ~20 lines of code.
- **Prompt caching calculator.** Given a prompt you plan to reuse N times, estimate the savings if you enabled Anthropic's prompt caching (Lesson 11). Compare cached vs. uncached total.
- **Batch API comparison.** Given a batch of prompts, show the savings if run through a Batch API (50% off on OpenAI and Anthropic).
- **Comparison with actual API calls.** Add a `--verify` flag that calls a cheap model and compares your estimated cost vs. the actual billed tokens. Should match within 1-2%.

---

## Starter scaffold

```python
# token_economics/tokenizers.py
from dataclasses import dataclass
import tiktoken
from anthropic import Anthropic

@dataclass
class TokenCount:
    model_name: str
    tokenizer: str
    count: int

_anthropic = Anthropic()

def count_tiktoken(text: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def count_anthropic(text: str, model: str = "claude-sonnet-4-6") -> int:
    resp = _anthropic.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens

def count_all(text: str, tokenizer_specs: list[tuple[str, str]]) -> list[TokenCount]:
    results = []
    for model_name, tokenizer in tokenizer_specs:
        if tokenizer.startswith("tiktoken_"):
            encoding = tokenizer.removeprefix("tiktoken_")
            count = count_tiktoken(text, encoding)
        elif tokenizer == "anthropic":
            count = count_anthropic(text)
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer}")
        results.append(TokenCount(model_name, tokenizer, count))
    return results
```

```python
# token_economics/pricing.py
import yaml
from pydantic import BaseModel
from pathlib import Path

class ModelPricing(BaseModel):
    name: str
    provider: str
    input_per_mtok: float
    output_per_mtok: float
    context_window: int
    tokenizer: str

class PricingConfig(BaseModel):
    models: list[ModelPricing]

def load_pricing(path: Path = Path("pricing.yaml")) -> PricingConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return PricingConfig(**data)

def estimate_cost(
    pricing: ModelPricing,
    input_tokens: int,
    output_tokens: int,
) -> float:
    return (
        input_tokens * pricing.input_per_mtok
        + output_tokens * pricing.output_per_mtok
    ) / 1_000_000
```

```python
# token_economics/cli.py
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from .pricing import load_pricing, estimate_cost
from .tokenizers import count_all

app = typer.Typer(help="Token economics CLI — count tokens and estimate costs.")
console = Console()

def read_input(text: str | None, file: Path | None) -> str:
    if text:
        return text
    if file:
        return file.read_text()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    typer.echo("No input provided. Use --text, --file, or pipe via stdin.", err=True)
    raise typer.Exit(1)

@app.command()
def count(
    text: str = typer.Argument(None),
    file: Path = typer.Option(None, "--file", "-f"),
):
    """Show token counts across tokenizers."""
    content = read_input(text, file)
    pricing = load_pricing()
    tokenizer_specs = [(m.name, m.tokenizer) for m in pricing.models]
    counts = count_all(content, tokenizer_specs)

    table = Table(title="Token counts")
    table.add_column("Model")
    table.add_column("Tokenizer")
    table.add_column("Tokens", justify="right")
    for c in counts:
        table.add_row(c.model_name, c.tokenizer, str(c.count))
    console.print(table)

@app.command()
def cost(
    text: str = typer.Argument(None),
    file: Path = typer.Option(None, "--file", "-f"),
    output_ratio: float = typer.Option(0.5, "--output-ratio"),
    budget: float = typer.Option(None, "--budget"),
):
    """Estimate cost per request across models."""
    content = read_input(text, file)
    pricing = load_pricing()
    rows = []
    for m in pricing.models:
        # Use the model's own tokenizer for its count
        tc = count_all(content, [(m.name, m.tokenizer)])[0]
        output_tokens = int(tc.count * output_ratio)
        total_cost = estimate_cost(m, tc.count, output_tokens)
        over_window = tc.count > m.context_window
        rows.append((m, tc.count, output_tokens, total_cost, over_window))

    rows.sort(key=lambda r: r[3])   # cheapest first

    table = Table(title="Estimated cost per request")
    table.add_column("Model")
    table.add_column("In tok", justify="right")
    table.add_column("Out tok", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Notes")

    for m, in_tok, out_tok, total, over in rows:
        notes = []
        if over: notes.append("[red]over context[/red]")
        if budget and total > budget: notes.append("[yellow]over budget[/yellow]")
        table.add_row(
            m.name,
            str(in_tok),
            str(out_tok),
            f"${total:.5f}",
            " ".join(notes),
        )
    console.print(table)

if __name__ == "__main__":
    app()
```

Install as an editable CLI:

```toml
# pyproject.toml
[project]
name = "token-economics"
version = "0.1.0"
dependencies = ["tiktoken", "anthropic", "typer", "rich", "pyyaml", "pydantic"]

[project.scripts]
token-econ = "token_economics.cli:app"
```

```bash
pip install -e .
echo "Hello, world!" | token-econ count
```

---

## Evaluation rubric — how to know you're done

- [ ] CLI runs and all four commands work (`count`, `cost`, `batch`, `models`)
- [ ] At least 2 tokenizers implemented and giving different counts for the same input
- [ ] At least 5 models in the pricing config, across at least 2 providers
- [ ] Pricing is loaded from a YAML/TOML file — not hardcoded in Python
- [ ] Handles stdin, `--file`, and direct text input
- [ ] `--budget` correctly filters and highlights
- [ ] At least one unit test for tokenizer wrapper and one for cost calculation
- [ ] Tested with: empty string, 10-token prompt, 5000-token prompt, one Chinese prompt, one code block, one JSON document
- [ ] README shows 3-5 example commands and their output
- [ ] Installable via `pip install -e .` with a `token-econ` entry point
- [ ] You've used the tool at least once on a real prompt from Project 1 and noted the result in your README

---

## Common pitfalls

- **Hardcoding pricing in Python.** You will forget to update it. Put it in a config file from day one.
- **Mixing up `tiktoken` encodings.** `cl100k_base` is GPT-3.5/4. `o200k_base` is GPT-4o/GPT-5. Using the wrong one gives you the wrong count. Always check with `tiktoken.encoding_for_model("gpt-5.4")` first.
- **Using `tiktoken` for Claude or Llama.** It's wrong. The counts will be close but not identical. Use the Anthropic count_tokens endpoint for Claude. For Llama, load the real tokenizer from `transformers`.
- **Confusing input cost with total cost.** Total cost = input + output. Don't forget the output half. If you report only input cost, you're lying to yourself about budgets.
- **Treating all text as English.** Run your tool on Chinese or Arabic text and observe the count difference. Your users will.
- **Not handling the empty-input case.** `tiktoken.encode("")` returns `[]` — 0 tokens. Your cost calculation divides by nothing gracefully, but your ranking may fail. Add a guard.
- **Forgetting `pip install -e .`** and running the CLI from inside the source directory with `python -m`. Install it as a proper CLI — half the value is being able to pipe to it.
- **Token counts that don't match your Project 1 API responses.** They should match within ~1%. If they don't, you're either counting wrong or comparing the wrong tokenizer to the wrong model.

---

## Cost estimate

Near-zero. The Anthropic count_tokens endpoint is free (as of early 2026 — check current pricing). `tiktoken` is entirely local. The only real cost is if you add the stretch "verify against actual API" feature, which would burn a few cents.

**Budget: $0-1.**

---

## What to deliver

```
token-economics/
├── pyproject.toml
├── token_economics/
│   ├── __init__.py
│   ├── cli.py
│   ├── tokenizers.py
│   ├── pricing.py
│   └── models.py
├── pricing.yaml
├── tests/
│   └── test_tokenizers.py
├── README.md             ← with example usage + screenshots of rich output
└── .gitignore
```

Your README should include:
- The value proposition in one sentence
- Install instructions
- 3-5 example invocations with expected output
- Where to edit pricing when it changes
- Known limitations (e.g. "Llama tokenization is approximated with tiktoken")

---

## Going further (after you finish)

- Add the **cache savings calculator** for Anthropic prompt caching (Lesson 11). Given "this prompt, reused N times," compute the real savings.
- Add a **budget tracker**: every time you run the tool, append to a local SQLite log. At the end of the month, print "if you'd actually run these, you would have spent $X".
- Ship it as a **VS Code extension** — highlight a block of text, see its token count in the status bar. The ultimate personal developer tool.
- Publish it to **PyPI** so other people can `pip install token-economics`. Don't forget the license.

---

## References

- OpenAI, *tiktoken library*. https://github.com/openai/tiktoken
- OpenAI Cookbook, *How to count tokens with tiktoken*. https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
- Anthropic, *Token counting endpoint*. https://docs.claude.com/en/docs/build-with-claude/token-counting
- Hugging Face, *Tokenizer summary* (background on BPE, WordPiece, SentencePiece). https://huggingface.co/docs/transformers/tokenizer_summary
- Typer, *CLI framework docs*. https://typer.tiangolo.com/
- Rich, *Python rich library for terminal output*. https://rich.readthedocs.io/

---

[← Previous Project](01-model-comparison.md) | [Back to LLM Fundamentals](../README.md) | [Next Project → Mini Transformer](03-mini-transformer.md)
