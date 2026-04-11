# Build Your Own Agent Harness

🔴 **Capstone-tier**

Prerequisite: complete the three teardowns in [harnesses/README.md](../harnesses/README.md) first. Do not start this project cold.

Everyone can build an agent loop in an afternoon. Almost nobody can build a harness you'd trust with your own repo. This is the project that forces every design decision a real harness team makes — and it will change how you read Claude Code's and Codex's source forever.

## What You'll Build

A terminal-based coding agent harness — call it `mini-harness` — that can read files, write files, run shell commands, and collaborate on a codebase through multi-turn conversation. It must ship with a permission system, a hook lifecycle, a tool registry, context compaction, session persistence, and subagent dispatch. It doesn't need to be fast or pretty. It needs to be *designed correctly*.

You are not trying to beat Claude Code. You are trying to understand Claude Code by rebuilding its skeleton.

## What You'll Learn

- How tool registries, permission classes, and rendering hints fit together
- Why context compaction is the hardest problem in the outer loop
- How hook lifecycles are actually implemented (spoiler: it's mostly string formatting and subprocess calls)
- The difference between "the model wants to read a file" and "the user sees a file read happen"
- Subagent dispatch as a context-protection mechanism, not a feature
- Why every harness eventually invents its own session format
- All the implicit decisions the Claude Agent SDK makes for you

## Tech Stack

- Python 3.11+ OR TypeScript/Node 20+ — pick the one you're more fluent in
- `anthropic` SDK (or `openai` — either is fine; Anthropic maps more cleanly to these concepts)
- `rich` (Python) or `ink` (Node) for the TUI — or plain stdout if you want to focus on mechanics
- `sqlite3` for session persistence
- A scratch repo to test against (clone any small open-source project)

**Do not use** the Claude Agent SDK, `langgraph`, `openai-agents`, or any agent framework. The whole point is to build the layer those libraries abstract away.

## Requirements

### Core (must have)

1. **Agent loop**
   - Multi-turn conversation with a model
   - Streaming output to the terminal
   - Graceful handling of tool-call → tool-result cycles

2. **Tool registry** with at least these tools:
   - `read_file(path)` — returns file contents
   - `write_file(path, content)` — writes a file
   - `edit_file(path, old, new)` — exact-string edit
   - `bash(command)` — runs a shell command, captures stdout/stderr
   - `glob(pattern)` — lists matching files
   - Each tool must declare: schema, permission class, error contract

3. **Permission system** with at least three classes:
   - `safe` — runs without prompting (e.g., `read_file`, `glob`)
   - `needs_approval` — prompts the user before executing (e.g., `write_file`, `edit_file`, most `bash`)
   - `destructive` — requires explicit confirmation with the exact command echoed back (e.g., `rm -rf`, `git reset --hard`)
   - Must support "allow once" and "allow for session" responses
   - Must have a `--yolo` mode that skips prompts (for demo purposes — and to feel the horror of running it)

4. **Hook lifecycle** — at least these events, each firing a user-defined shell command with a JSON payload on stdin:
   - `SessionStart`
   - `UserPromptSubmit`
   - `PreToolUse`
   - `PostToolUse`
   - `Stop`
   - Hooks are configured via a `harness.config.json` file
   - A hook can *block* a tool call by exiting non-zero (this is the killer feature — make sure it works)

5. **Context management**
   - Track token usage per turn
   - At a configurable threshold, trigger compaction: summarize old turns and replace them with the summary
   - Always preserve the original user request and the last N turns verbatim
   - Tool results longer than a threshold get truncated with a `[truncated, N bytes]` marker

6. **Session persistence**
   - Every message, tool call, and tool result is written to a SQLite DB as it happens
   - `mini-harness --resume <session-id>` restores a session exactly
   - `mini-harness --fork <session-id>` creates a new session branched from an existing one

7. **Subagent dispatch**
   - A `dispatch_agent(task, tools)` tool that spawns a child agent with:
     - A fresh context
     - A restricted tool subset (e.g., read-only)
     - Its own session record
   - The parent receives only the child's final summary, not its full transcript
   - Demonstrate context savings: run the same task with and without dispatch, compare token usage

### Stretch goals (pick two)

- **Slash commands** — user-defined prompt templates stored in `~/.mini-harness/commands/` invoked as `/name`
- **Skills / progressive disclosure** — a skill is a Markdown file with frontmatter; when the user mentions a keyword, the skill is lazily loaded into the system prompt
- **MCP client** — connect to an external MCP server and expose its tools through the harness's tool registry (this proves you understand *both* sides of the MCP boundary)
- **Worktree isolation** — any tool call that writes files runs in a `git worktree` that the user can inspect and merge back
- **Parallel subagents** — dispatch multiple subagents concurrently with `asyncio.gather` / `Promise.all` and display their status in the TUI

## Architecture Deliverable

Before writing code, produce a diagram (draw.io, Excalidraw, or Mermaid) showing:

- The outer loop and inner loop boundaries
- Where the permission check sits relative to the tool call
- Where each hook fires
- How the context compactor reads from and writes to the session DB
- How a subagent dispatch relates to the parent's context

Commit the diagram to the repo as `ARCHITECTURE.md`. You'll update it as your understanding sharpens.

## Evaluation — How You Know You're Done

Run your harness against these five real tasks in a throwaway clone of any small open-source repo. Record the session, note friction points.

1. **Read-only exploration:** "What does this codebase do? Summarize the architecture."
2. **Simple edit:** "Find the README and add a one-line disclaimer at the top."
3. **Destructive prompt:** "Delete all the test files." — your harness must force approval with the exact file list.
4. **Long task:** "Refactor module X to extract Y into a separate file, then update all imports." — should trigger compaction at least once.
5. **Dispatched research:** "Find every place that uses function F, summarize how it's used, then propose a new signature." — should use a subagent for the search.

For each task, answer: did the harness do the right thing? Did the hooks fire? Did compaction work? What did you have to fix?

## Hints

- **Start with the tool registry and permissions.** Not the TUI, not streaming, not hooks. Tools and permissions are the hardest part to retrofit.
- **Hooks are just subprocess calls with JSON on stdin.** Don't over-engineer — Claude Code's implementation is ~300 lines.
- **For compaction, ask the model to summarize the old turns.** Don't try to write a deterministic summarizer — you'll lose signal.
- **Subagent dispatch is a tool like any other.** The "magic" is that its implementation creates a new session and runs the full agent loop recursively.
- **Session persistence should be append-only.** Never mutate past rows. This makes fork and resume trivial.
- **Write a real config file format from day one** — JSON or TOML. Don't hardcode things you'll want to change.
- **Test against your own repo early.** You'll discover permission bugs faster than any synthetic test.

## Cost Estimate

- Building: ~$30-60 in API credits across all the iteration and debugging
- Final evaluation (5 tasks): ~$10-15
- Budget $100 total to be safe. Use Haiku for dev loops, Sonnet for real runs.

## What to Publish

- Source code on GitHub with a clear README
- `ARCHITECTURE.md` with your diagrams
- A blog post or write-up: "What I learned building my own Claude Code" — compare your design decisions against Claude Code, Codex, and Aider
- A short demo video (even a raw terminal recording is fine)

This project is the strongest possible signal for an AI engineering role. It proves you understand agents at the systems level, not just the SDK level.

---

[← Back to Agents](../README.md) | [Harnesses Sub-Section](../harnesses/README.md)
