# 4 — Advanced Agents

**Goal:** Build tool-using agents, implement MCP, and orchestrate multi-agent systems.

## Learning Objectives

- Build agents that use tools via function calling (search, code execution, file I/O)
- Implement the Model Context Protocol (MCP) for standardized tool integration
- Design and orchestrate multi-agent systems with planning, execution, and review

---

## 📚 Readings & Resources

| Resource | Type | Why |
|----------|------|-----|
| [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) | Blog | **Must-read** — the definitive guide to agent architecture patterns |
| [Anthropic: Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Blog | How to manage context for long-running agents |
| [Model Context Protocol (MCP) Spec](https://modelcontextprotocol.io/) | Docs | The emerging standard for tool integration |
| [MCP vs A2A: Complete Guide](https://dev.to/pockit_tools/mcp-vs-a2a-the-complete-guide-to-ai-agent-protocols-in-2026-30li) | Blog | Comparison of the two agent protocol standards |
| [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | Blog | Foundational post on agent architectures (planning, memory, tools) |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Docs | Official agent SDK with handoffs and tool use |
| [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/) | Docs | State-machine agent orchestration |

---

## 🔨 Projects

| # | Project | Difficulty | Description |
|---|---------|------------|-------------|
| 1 | [Tool-Using Chatbot](projects/01-tool-using-chatbot.md) | 🟢 Beginner | Chatbot with web search, calculator, and file tools |
| 2 | [MCP Server + Client](projects/02-mcp-server.md) | 🟡 Intermediate | SQLite exposed as MCP tools with an agent client |
| 3 | [Multi-Agent Research System](projects/03-multi-agent-research.md) | 🟠 Advanced | Planner + researchers + critic + synthesizer |

---

## Key Concepts

After completing this section, you should understand:

- The agent loop: observe → think → act → observe
- Function calling as the mechanism for tool use
- How to design tool interfaces (clear names, descriptions, schemas)
- Error handling and retry strategies for tool execution
- The Model Context Protocol (MCP) and why standards matter
- Multi-agent patterns: delegation, debate, pipeline, hierarchical
- Memory management for long-running agent conversations
- When to use agents vs. simpler approaches (not everything needs an agent)

---

[← RAG](../03-rag/) | [Home](../README.md) | [Next → Observability](../05-observability/)
