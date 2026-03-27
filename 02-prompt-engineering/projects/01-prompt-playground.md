# Prompt Playground

🟢 **Beginner**

Every AI engineer needs a prompt testing tool. Build your own — it's more useful than any third-party playground because you control it.

## What You'll Build

A CLI tool that sends prompts to multiple providers (OpenAI, Anthropic, Groq), lets you tweak parameters like temperature and system prompts, and saves results for comparison.

## What You'll Learn

- Working with multiple LLM provider APIs
- Understanding how parameters affect generation
- Building reusable developer tools
- Saving and comparing experiment results

## Tech Stack

- Python 3.11+
- `openai`, `anthropic`, `groq` SDKs
- `click` or `typer` for CLI
- JSON or SQLite for result storage

## Requirements

- Send prompts to at least 3 providers from one command
- Configurable parameters: temperature, top-p, max tokens, system prompt
- Display responses side-by-side with model names
- Save each run (prompt, params, responses, timestamps) to a local store
- Replay previous prompts with different parameters
- Support loading prompts from files (for multi-line prompts)
- Show token usage and estimated cost per response
- Support streaming output (print tokens as they arrive)
- Config file for API keys (don't hardcode them)

## Stretch Goals

- Add a "diff" mode that highlights differences between model responses
- Build a simple web UI with Gradio
- Support prompt templates with `{variable}` substitution

## Hints

- Use environment variables or a `.env` file for API keys — `python-dotenv` makes this easy
- Streaming APIs differ by provider — abstract the streaming interface so you can add new providers later
- Start with synchronous calls, then add async for parallel requests across providers

## Cost Estimate

~$1-2 in API credits. Use Groq (free) for most iteration.

---

[← Back to Prompt Engineering](../README.md)
