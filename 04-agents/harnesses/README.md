# Deep Dive — Agent Harnesses

**Goal:** Understand what a production agent harness actually is, how the major ones are built, and what design decisions separate a toy agent loop from a tool people trust on their filesystem.

> An *agent* is a loop. A *harness* is everything around the loop that makes it safe, steerable, and useful for real work — permissions, context management, hooks, tool registries, session state, slash commands, subagents, sandboxing. This is the gap between "I built an agent" and "I shipped one."

---

## Why This Section Exists

The earlier part of `04-agents` teaches you the agent loop, function calling, MCP, and multi-agent orchestration. Those are the *mechanics*. But every shipped coding agent — Claude Code, OpenAI Codex CLI, Cursor's agent mode, Aider, Continue, Cline — wraps those mechanics in a substantial piece of software that handles the parts you don't see in a tutorial:

- What happens when the model wants to `rm -rf /`?
- How does the agent remember what it did 200 turns ago without blowing out its context window?
- Who decides when a tool call needs human approval and when it doesn't?
- How do you plug in custom tools, custom workflows, custom prompts — without forking the whole thing?
- How does one agent spawn another, and how do they share (or isolate) state?

These questions are the harness's job. Studying them teaches you more about applied agent design than any single tutorial will.

---

## 📚 Readings & Resources

| Resource | Type | Why |
|----------|------|-----|
| [Claude Code Overview](https://docs.claude.com/en/docs/claude-code/overview) | Docs | The reference harness — read the whole docs tree |
| [Claude Code: Hooks](https://docs.claude.com/en/docs/claude-code/hooks) | Docs | Lifecycle events: PreToolUse, PostToolUse, SessionStart, etc. |
| [Claude Code: Skills](https://docs.claude.com/en/docs/claude-code/skills) | Docs | Progressive disclosure of capabilities via on-demand instructions |
| [Claude Code: Subagents](https://docs.claude.com/en/docs/claude-code/sub-agents) | Docs | Dispatching isolated agents to protect the main context |
| [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview) | Docs | Build your own harness on Anthropic's primitives |
| [OpenAI Codex CLI (GitHub)](https://github.com/openai/codex) | Code | Read the source — especially the tool + permission layers |
| [Aider (GitHub)](https://github.com/Aider-AI/aider) | Code | The OG terminal coding agent — cleaner, smaller codebase to study |
| [Cline (GitHub)](https://github.com/cline/cline) | Code | VS Code extension harness — different UX constraints than CLI |
| [Anthropic: Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Blog | The theory behind why harnesses compact, summarize, and dispatch |
| [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) | Blog | Re-read with harness eyes — which patterns each harness implements |

---

## Core Concepts

Work through these in order. Each one is a design decision that every harness makes — sometimes implicitly.

### 1. The Outer Loop vs the Inner Loop

Every harness has two loops:

- **Inner loop:** one turn of the agent — send messages, receive tool calls, execute them, send results back.
- **Outer loop:** session management — when to compact context, when to persist state, when to surface to the user, when to stop.

Toy agents only have the inner loop. Harnesses live or die on the outer loop.

### 2. Tool Registry & Tool Contracts

Tools aren't just "functions the model can call." A harness tool carries:

- A schema (inputs/outputs)
- A permission class (safe / needs-approval / destructive)
- A rendering hint (how the tool call appears in the UI)
- An error contract (what happens on failure — retry, surface, abort)
- Optional pre/post hooks

Compare how Claude Code, Codex, and Aider each model these.

### 3. Permission Models & Sandboxing

The hardest design problem in a coding harness. Options include:

- **Allowlist/denylist** of commands (Codex's approach)
- **Permission modes** with escalation prompts (Claude Code's `default` / `acceptEdits` / `plan` / `bypassPermissions`)
- **Filesystem sandboxing** via worktrees or containers
- **Network sandboxing** (can the agent `curl`?)
- **Session-scoped vs durable allowances** (`"allow this once" vs "always allow in this repo"`)

Understand why each harness picked what it picked.

### 4. Context Management

The outer loop's hardest job. Techniques you'll see:

- **Compaction** — summarize old turns to free tokens
- **Progressive disclosure** — load skills/docs only when relevant
- **Subagent dispatch** — run a sub-task in a fresh context, return only the summary
- **External memory** — persist facts to disk so they survive compaction
- **Selective tool-result truncation** — keep the tool call, drop the 5000-line stdout

### 5. Hook Lifecycle

Hooks are user-defined shell commands that fire on lifecycle events. They let users extend the harness without modifying it. Study Claude Code's hook events — `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `SessionStart`, `Stop`, `SubagentStop` — and think about *why each exists*.

### 6. Slash Commands, Skills, and Custom Workflows

How do users teach the harness new tricks without shipping a new release?

- **Slash commands** — user-defined prompts invoked by name (`/commit`, `/review-pr`)
- **Skills** — bundles of instructions the model loads on demand
- **Custom agents / modes** — pre-configured personas with their own tools and system prompts

### 7. Subagent Dispatch

The newest pattern and arguably the most important one. Dispatching an isolated agent lets the parent:

- Protect its own context from large tool outputs
- Run N tasks in parallel
- Use a cheaper model for a narrow job
- Enforce a different permission policy (e.g., read-only research agent)

This is how harnesses scale to long sessions without drowning in their own transcript.

### 8. Session State & Resumability

Where does the conversation live? How do you resume after a crash? How do you fork a session to try a different approach? Every harness answers these differently; most answer poorly.

---

## Teardown Exercises

Before attempting the project, complete these three teardowns. Write each one as a short document (1-2 pages) in your notes.

### Teardown 1 — Claude Code

Read the docs tree end-to-end. Then answer:

1. What are *all* the lifecycle events a hook can fire on? Why does each exist?
2. How does a subagent dispatch differ from a normal tool call, from the main agent's perspective?
3. What does `acceptEdits` permission mode allow vs `default`? What's the threat model?
4. How do Skills differ from slash commands? When would you pick one over the other?
5. How does Claude Code compact context, and what signals trigger it?

### Teardown 2 — OpenAI Codex CLI

Clone [openai/codex](https://github.com/openai/codex). Read the source. Then answer:

1. Where is the tool registry defined? How does it compare to Claude Code's?
2. How does Codex decide whether a shell command needs approval?
3. What's in the system prompt, and how does it change based on mode?
4. How does Codex handle session persistence and resumption?
5. What's the smallest change you'd make to add a hook system like Claude Code's?

### Teardown 3 — Aider

Clone [Aider-AI/aider](https://github.com/Aider-AI/aider). This codebase is smaller and easier to read than Codex. Then answer:

1. How does Aider represent "the files in the conversation"? How does that differ from Claude Code's file reading?
2. How does Aider handle edits — does it diff, patch, or rewrite?
3. What's Aider's permission model (if any)? What happens if the model proposes a destructive edit?
4. How does Aider decide what to include in the context window as the conversation grows?
5. What trade-offs has Aider made that the others haven't?

---

## 🔨 Project

After the teardowns, you're ready for the main event:

→ **[Project 4: Build Your Own Harness](../projects/04-build-your-own-harness.md)** 🔴 Capstone-tier

---

## Key Concepts (Checklist)

After this sub-section, you should be able to explain:

- [ ] The difference between an agent loop and an agent harness
- [ ] Why every shipped coding agent ends up implementing a permission model
- [ ] At least three context management strategies and when each applies
- [ ] The hook lifecycle and why users want extension points at specific moments
- [ ] How subagent dispatch protects the parent's context window
- [ ] The trade-offs between slash commands, skills, and custom agents
- [ ] How session state is persisted and resumed (and why most harnesses do it poorly)
- [ ] Why "just wrap the SDK" is not enough to build a production harness

---

[← Back to Agents](../README.md) | [Home](../../README.md)
