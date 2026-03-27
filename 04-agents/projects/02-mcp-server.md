# MCP Server + Client

🟡 **Intermediate**

The Model Context Protocol (MCP) is becoming the standard for how AI agents discover and use tools. Build a server that exposes a database as tools, then build an agent that talks to it.

## What You'll Build

An MCP server that exposes a SQLite database as tools (query, insert, schema inspection), plus an agent client that connects to it. Ask "How many orders were placed last week?" and the agent writes and runs SQL under the hood.

## What You'll Learn

- The Model Context Protocol specification
- Building MCP-compliant servers
- Tool discovery and invocation patterns
- Natural language to SQL translation
- Client-server architecture for AI tools

## Tech Stack

- Python 3.11+
- `mcp` Python SDK
- SQLite
- `openai` or `anthropic` SDK for the agent
- A sample database (create one with realistic data)

## Requirements

- Build an MCP server that exposes these tools:
  - `list_tables` — show all tables and their schemas
  - `describe_table` — show columns, types, and sample rows for a table
  - `query` — execute a read-only SQL query and return results
  - `insert` — insert a row into a specified table
- Follow the MCP spec for tool definitions (name, description, input schema)
- Implement proper error handling (invalid SQL, table not found, etc.)
- Build an agent client that:
  - Connects to the MCP server
  - Discovers available tools automatically
  - Takes natural language questions and translates them to tool calls
  - Presents results in a human-readable format
- Create a sample database with at least 3 related tables (e.g., orders, customers, products)
- Seed it with realistic sample data (100+ rows)
- The agent should handle multi-step queries ("What's the average order value for customers in Singapore?")

## Stretch Goals

- Add a `create_table` tool and let the agent create new tables from natural language descriptions
- Implement query safety checks (prevent DROP TABLE, etc.)
- Add a second MCP server (e.g., a file system server) and have the agent use both

## Hints

- The MCP Python SDK handles the protocol plumbing — focus on defining good tool schemas
- For natural language → SQL, a well-crafted system prompt with the schema works better than you'd expect
- Include the table schema in the system prompt so the LLM doesn't have to call `describe_table` every time

---

[← Back to Agents](../README.md)
