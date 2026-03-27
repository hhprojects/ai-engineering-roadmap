# Tool-Using Chatbot

🟢 **Beginner**

An LLM that can only talk is limited. Give it tools — web search, a calculator, file access — and it becomes genuinely useful. This is the foundation of every AI agent.

## What You'll Build

A chatbot with 3-4 tools that it can invoke via function calling. It decides when to use which tool, handles errors gracefully, and maintains conversation history across turns.

## What You'll Learn

- Function calling / tool use with LLM APIs
- Designing tool interfaces with clear schemas
- Implementing the agent loop (query → tool call → result → response)
- Error handling for tool execution
- Managing conversation memory

## Tech Stack

- Python 3.11+
- `openai` or `anthropic` SDK (both support function calling)
- `serpapi` or `tavily` for web search (free tiers available)
- `click` or `typer` for CLI

## Requirements

- Implement at least 4 tools:
  - **Web search** — search the internet via SerpAPI or Tavily (free tier)
  - **Calculator** — evaluate math expressions safely
  - **File reader** — read content from local files
  - **Weather** — get current weather for a location (free API)
- Define each tool with a clear name, description, and JSON schema
- The LLM decides which tool to use (or no tool) based on the user's message
- Handle tool errors gracefully (API failures, file not found, etc.)
- Implement retry logic for transient failures
- Maintain conversation history (the chatbot remembers previous messages)
- Display tool calls and results in the conversation (transparency)
- Interactive CLI loop (type messages, see responses, ctrl-C to exit)

## Stretch Goals

- Add a "code execution" tool that runs Python code in a sandbox
- Implement multi-step tool use (LLM can chain multiple tool calls in one turn)
- Add conversation persistence (save/load chat history to disk)

## Hints

- Start with one tool (calculator), get the agent loop working, then add more
- The tool descriptions matter as much as the tool implementations — unclear descriptions confuse the LLM
- Don't use `eval()` for the calculator — use `ast.literal_eval()` or a safe math parser

## Cost Estimate

~$2-3 in API credits for development and testing.

---

[← Back to Agents](../README.md)
