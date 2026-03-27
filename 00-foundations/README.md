# §0 — Software Engineering Foundations

**Goal:** Ensure Python proficiency, backend basics, and DevOps fundamentals before touching AI.

## Learning Objectives

- Write clean, typed, tested Python with modern tooling
- Build and document REST APIs with FastAPI
- Containerize applications with Docker and manage them with git

---

## 📚 Readings & Resources

| Resource | Type | Why |
|----------|------|-----|
| [Python Official Tutorial](https://docs.python.org/3/tutorial/) | Docs | The canonical source — async, decorators, typing |
| [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) | Docs | Official tutorial, best-in-class for learning modern Python APIs |
| [Docker Getting Started](https://docs.docker.com/get-started/) | Docs | Official Docker walkthrough |
| [Git Immersion](https://gitimmersion.com/) | Tutorial | Hands-on git learning |
| [Corey Schafer: Python Tutorials](https://www.youtube.com/playlist?list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU) | 🎬 YouTube | Excellent Python fundamentals playlist |
| [TechWorld with Nana: Docker Tutorial](https://www.youtube.com/watch?v=3c-iBn73dDE) | 🎬 YouTube | Best Docker crash course on YouTube |

---

## 🔨 Projects

| # | Project | Difficulty | Description |
|---|---------|------------|-------------|
| 1 | [CLI Task Manager](projects/01-cli-task-manager.md) | 🟢 Beginner | Python CLI todo app with SQLite and pytest |
| 2 | [REST API with FastAPI](projects/02-fastapi-bookmarks.md) | 🟡 Intermediate | Bookmark/notes API with auth and Docker |
| 3 | [Async Data Pipeline](projects/03-async-data-pipeline.md) | 🟠 Advanced | Concurrent web scraper with retries and structured logging |

---

## Key Concepts

After completing this section, you should understand:

- Python type hints and when to use them
- How to structure a Python project (packages, modules, `__init__.py`)
- Writing and running tests with pytest
- Building REST APIs with request/response models (Pydantic)
- SQL basics with SQLite
- Docker images, containers, volumes, and Dockerfiles
- Git branching, committing, and collaboration workflows
- Async/await patterns in Python

---

[Home](../README.md) | [Next → LLM Fundamentals](../01-llm-fundamentals/)
