# Async Data Pipeline

🟠 **Advanced**

Build an async web scraper that fetches pages concurrently, extracts structured data, and handles the real-world messiness of network programming — retries, rate limits, and failures.

## What You'll Build

A concurrent data pipeline that scrapes web pages using `aiohttp`, extracts structured data with CSS selectors or regex, stores results in SQLite, and handles all the edge cases production scrapers need.

## What You'll Learn

- Async/await patterns with `asyncio`
- Concurrent HTTP requests with `aiohttp`
- Rate limiting and retry strategies
- Structured logging for debugging async code
- Docker containerization of async applications

## Tech Stack

- Python 3.11+
- `aiohttp` for async HTTP
- `beautifulsoup4` for HTML parsing
- `sqlite3` or `aiosqlite`
- `structlog` for structured logging
- Docker
- pytest + `pytest-asyncio`

## Requirements

- Accept a list of URLs (from file or CLI args)
- Fetch pages concurrently with configurable concurrency limit (semaphore)
- Extract structured data using configurable selectors
- Store results in SQLite with timestamps and source URLs
- Implement exponential backoff retry (3 attempts, 1s → 2s → 4s)
- Rate limit requests (e.g., max 5 requests/second per domain)
- Handle common failures gracefully: timeouts, 404s, 429s, connection errors
- Structured JSON logging with `structlog`
- Dockerized with a `Dockerfile`
- At least 15 tests (mock HTTP responses with `aioresponses`)

## Stretch Goals

- Add a progress bar with `tqdm` or `rich`
- Support resumable scraping (skip already-fetched URLs on restart)
- Add a simple CLI dashboard showing success/failure counts in real-time

## Hints

- `asyncio.Semaphore` is your friend for controlling concurrency — don't fire 1000 requests at once
- Test with `aioresponses` to mock HTTP calls — don't hit real servers in tests
- Structure your code as pipeline stages: fetch → parse → store. Each stage can be tested independently.

---

[← Back to Foundations](../README.md)
