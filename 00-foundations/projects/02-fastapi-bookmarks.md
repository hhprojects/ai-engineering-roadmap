# REST API with FastAPI

🟡 **Intermediate**

Build a bookmark/notes API with full CRUD, authentication, and Docker support. This is the kind of backend you'll build AI features on top of later.

## What You'll Build

A REST API for saving and organizing bookmarks/notes, complete with Pydantic models, API key auth, auto-generated OpenAPI docs, and a Docker container you can deploy anywhere.

## What You'll Learn

- Building REST APIs with FastAPI
- Request/response modeling with Pydantic
- API authentication patterns
- Docker containerization
- Writing integration tests for APIs

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- SQLite (via `aiosqlite` or standard `sqlite3`)
- Pydantic v2
- Docker
- pytest + `httpx` for testing

## Requirements

- Full CRUD for bookmarks (create, read, update, delete)
- Each bookmark has: URL, title, description, tags, created_at
- Filter bookmarks by tag, search by title/description
- Pydantic models for all request/response schemas
- API key authentication (header-based)
- Proper HTTP status codes (201 on create, 404 on not found, etc.)
- Auto-generated OpenAPI docs at `/docs`
- Dockerfile that builds and runs the app
- At least 15 tests covering happy paths and error cases
- Environment-based configuration (dev/prod settings)

## Stretch Goals

- Add pagination with cursor-based navigation
- Implement rate limiting with `slowapi`
- Add a simple HTML frontend served by FastAPI's static file support

## Hints

- FastAPI's dependency injection system is perfect for auth — create a `get_current_user` dependency
- Use `httpx.AsyncClient` with `app=app` for testing — no need to spin up a server
- Start with the Pydantic models, then the database layer, then the routes

---

[← Back to Foundations](../README.md)
