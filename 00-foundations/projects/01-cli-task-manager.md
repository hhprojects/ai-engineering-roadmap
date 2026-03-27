# CLI Task Manager

🟢 **Beginner**

Build a Python CLI todo app with SQLite, type hints, and pytest. This is your warm-up — practice the fundamentals you'll use in every project that follows.

## What You'll Build

A command-line task manager that stores tasks in SQLite, supports CRUD operations, and has a clean test suite. Think of it as `todo.txt` but backed by a database.

## What You'll Learn

- Python CLI development with `argparse` or `click`
- SQLite database operations
- Type hints and dataclasses
- Writing tests with pytest
- Project structure and packaging

## Tech Stack

- Python 3.11+
- `click` or `argparse` for CLI
- `sqlite3` (standard library)
- `pytest` for testing

## Requirements

- Add tasks with a title and optional due date
- List tasks with filtering (all, pending, completed)
- Mark tasks as complete
- Delete tasks
- Store everything in a local SQLite database
- Use type hints throughout
- Organize code into modules (not one giant file)
- Write at least 10 pytest tests covering core functionality
- Handle edge cases (empty titles, duplicate IDs, invalid dates)

## Stretch Goals

- Add priority levels (high/medium/low) with color-coded output
- Export tasks to JSON/CSV
- Add a `search` command with fuzzy matching

## Hints

- Start with the data model — define your Task dataclass first, then build the CLI around it
- Use `click.testing.CliRunner` for testing CLI commands without actually running the process
- SQLite's `datetime` type works well with Python's `datetime` module — no ORM needed

---

[← Back to Foundations](../README.md)
