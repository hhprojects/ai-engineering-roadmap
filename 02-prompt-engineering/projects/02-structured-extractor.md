# Structured Data Extractor

🟡 **Intermediate**

LLMs are incredibly good at extracting structured data from messy text — if you know how to ask. Build a tool that turns unstructured documents into clean JSON.

## What You'll Build

A tool that takes unstructured text (receipts, job listings, articles, emails) and extracts structured JSON using function calling and Pydantic models via the Instructor library. Handles edge cases, validation errors, and partial extractions.

## What You'll Learn

- Function calling / tool use APIs
- Pydantic models for schema enforcement
- The Instructor library for structured outputs
- Error handling and retry strategies for LLM extraction
- Building robust data pipelines with LLM components

## Tech Stack

- Python 3.11+
- `instructor` library
- `openai` or `anthropic` SDK
- Pydantic v2
- pytest

## Requirements

- Define Pydantic models for at least 3 different document types (e.g., receipts, job listings, contact info)
- Accept input as text (CLI arg, file, or stdin)
- Use Instructor + function calling to extract structured data
- Validate extracted data against the Pydantic schema
- Handle partial extractions gracefully (fill what you can, flag what's missing)
- Implement retry logic: if extraction fails validation, send the error back to the LLM for correction
- Support batch extraction (process multiple documents)
- Output clean JSON (both pretty-printed and JSONL for batch)
- Test suite with at least 20 diverse test inputs (messy, clean, edge cases)
- Track extraction success rate and cost per document

## Stretch Goals

- Add a "schema discovery" mode that infers the schema from example documents
- Support images (receipts, business cards) using vision APIs
- Build a simple evaluation harness comparing extraction accuracy across models

## Hints

- Instructor's `max_retries` parameter is powerful — it re-sends the validation error to the LLM automatically
- Start with the easiest document type (e.g., structured receipts) before tackling messy ones
- Use `Optional` fields in Pydantic models — not every document will have every field

---

[← Back to Prompt Engineering](../README.md)
