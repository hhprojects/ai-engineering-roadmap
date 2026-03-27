# Multi-Source RAG with Evaluation

🟠 **Advanced**

Real-world RAG doesn't just handle markdown files. Build a pipeline that ingests PDFs (with tables), web pages, and CSVs into a unified index — then prove it works with a proper evaluation harness.

## What You'll Build

A RAG system that handles multiple document types, stores them in a unified vector index, and includes a comprehensive evaluation framework. You'll create 50+ test questions, measure retrieval and generation quality, and use Langfuse for tracing.

## What You'll Learn

- Multi-format document ingestion (PDF, HTML, CSV)
- Table extraction from PDFs
- Building evaluation harnesses for RAG systems
- LLM-as-judge evaluation methodology
- Observability with Langfuse tracing
- Systematic experimentation (varying chunking strategies, measuring impact)

## Tech Stack

- Python 3.11+
- ChromaDB
- `pymupdf` or `pdfplumber` for PDF parsing
- `beautifulsoup4` for web pages
- `pandas` for CSV handling
- `fastembed` or OpenAI embeddings
- Langfuse (cloud free tier)
- pytest

## Requirements

- Ingest at least 3 document types: PDF (including ones with tables), web pages, and CSV files
- Extract tables from PDFs and convert to a searchable format
- Build a unified chunking pipeline that handles all source types
- Store source metadata (file type, page number, URL, etc.) with each chunk
- Create an evaluation dataset: 50+ question/answer pairs with expected source documents
- Measure retrieval metrics:
  - Hit rate (is the right document in top-k?)
  - Mean Reciprocal Rank (MRR)
- Measure generation metrics:
  - Answer faithfulness (is the answer grounded in retrieved context?)
  - Answer relevance (does it actually answer the question?)
- Use LLM-as-judge for faithfulness and relevance scoring
- Integrate Langfuse tracing for the full pipeline (ingest → retrieve → generate)
- Run experiments: compare at least 2 chunking strategies and show metric differences
- Generate an evaluation report (markdown or HTML)

## Stretch Goals

- Add automatic question generation from documents (use an LLM to create eval questions)
- Implement citation tracking — show which specific chunks were used in each answer
- Build a CI pipeline that runs evals on every change to the chunking or retrieval config

## Hints

- `pdfplumber` handles tables better than most PDF libraries — use `extract_tables()` for structured content
- For LLM-as-judge, use a simple rubric: "Score 1-5 on whether the answer is fully supported by the context." Average the scores.
- Langfuse's Python SDK decorators make tracing almost zero-effort — `@observe()` on your functions

## Cost Estimate

~$3-5 for evaluation runs (LLM-as-judge calls add up). Langfuse cloud free tier covers 50k observations/month.

---

[← Back to RAG](../README.md)
