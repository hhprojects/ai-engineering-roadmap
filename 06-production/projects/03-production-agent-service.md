# Production-Grade Agent Service

🟠 **Advanced**

This is the final boss. Deploy a multi-agent system with async task queues, streaming, security layers, monitoring, and load testing. If you can build this, you can build production AI.

## What You'll Build

A production-ready agent service with an async task queue (Celery + Redis), streaming responses via SSE, prompt injection detection, PII filtering, cost alerts, and a monitoring dashboard. Validated with load testing.

## What You'll Learn

- Async task queue architecture (Celery + Redis)
- Server-Sent Events (SSE) for streaming
- Prompt injection detection techniques
- PII filtering with regex and NER
- Cost monitoring and alerting
- Load testing with Locust
- Production monitoring and dashboarding

## Tech Stack

- Python 3.11+
- FastAPI
- Celery + Redis
- `openai` or `anthropic` SDK
- `presidio-analyzer` or regex for PII detection
- `locust` for load testing
- Docker + `docker-compose`
- Streamlit or Grafana for monitoring

## Requirements

- FastAPI service with endpoints for:
  - Submit agent task (returns task ID)
  - Stream task progress via SSE
  - Get task status and result
  - Admin: cost dashboard, active tasks
- Async task queue with Celery + Redis:
  - Agent tasks run as Celery tasks
  - Support task cancellation
  - Handle task timeouts (kill runaway agents)
- Security layer:
  - Prompt injection detection (keyword patterns + LLM classifier)
  - PII filtering on both input and output (emails, phone numbers, SSNs)
  - Input sanitization and length limits
- Cost management:
  - Track per-request and per-user costs
  - Alert (log/webhook) when daily spend exceeds configurable threshold
  - Budget caps per user
- Monitoring:
  - Request latency histograms
  - Error rates by type
  - Active task count
  - Simple dashboard (Streamlit or Grafana)
- Load testing:
  - Locust test file with realistic usage patterns
  - Test with 50 concurrent users
  - Document throughput and latency under load
- Full Docker Compose setup (API, worker, Redis, dashboard)
- At least 30 tests covering security, queueing, and streaming

## Stretch Goals

- Add WebSocket support alongside SSE
- Implement circuit breakers for LLM API calls
- Add A/B testing infrastructure (route % of traffic to different prompts)

## Hints

- Start with the happy path (task submission → execution → streaming result), then layer on security and monitoring
- For PII detection, `presidio-analyzer` is production-ready out of the box. Regex catches the obvious stuff; NER catches the rest.
- Locust is simple to set up — `locust -f locustfile.py` gives you a web UI for load testing

## Cost Estimate

~$5-10 for testing (many LLM calls during load testing). Use Groq for load testing to minimize cost.

---

[← Back to Production](../README.md)
