# Deploy Your RAG App

🟢 **Beginner**

Building locally is one thing. Getting it running in the cloud with proper configuration, health checks, and security is a different skill entirely. Time to deploy.

## What You'll Build

Take a previous project (your RAG app is ideal), containerize it with Docker, deploy to a free-tier cloud platform, and add the production basics: health checks, environment config, rate limiting, and API key auth.

## What You'll Learn

- Docker containerization for LLM applications
- Cloud deployment to free-tier platforms
- Environment-based configuration
- Basic API security (API keys, rate limiting)
- Health check patterns for AI services

## Tech Stack

- Python 3.11+
- Docker
- Railway, Render, or Fly.io (all have free tiers)
- FastAPI
- `slowapi` for rate limiting
- Your RAG project from 3

## Requirements

- Create a `Dockerfile` that builds and runs your app
- Use multi-stage builds to minimize image size
- All secrets (API keys) come from environment variables, never hardcoded
- Add a `/health` endpoint that checks:
  - API is responding
  - Vector database is accessible
  - LLM API key is valid (optional light check)
- Implement API key authentication (header-based)
- Add rate limiting with `slowapi` (e.g., 10 requests/minute per key)
- Deploy to one free-tier platform (Railway, Render, or Fly.io)
- Write a `README.md` with deployment instructions
- The deployed app should handle at least 5 concurrent requests without crashing
- Include a `.env.example` file documenting all required environment variables

## Stretch Goals

- Set up a CI/CD pipeline (GitHub Actions → auto-deploy on push to main)
- Add request logging with structured JSON logs
- Implement a simple API usage dashboard endpoint

## Hints

- Railway and Render both auto-detect Dockerfiles — push and deploy
- Use `gunicorn` with `uvicorn` workers for production: `gunicorn app:app -w 2 -k uvicorn.workers.UvicornWorker`
- Test your Docker image locally first: `docker build -t myapp . && docker run -p 8000:8000 --env-file .env myapp`

## Cost Estimate

Free — all platforms have free tiers sufficient for this project.

---

[← Back to Production](../README.md)
