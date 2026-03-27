# Multi-Agent Research System

🟠 **Advanced**

One agent is useful. A team of agents that plan, research, critique, and synthesize? That's how you tackle complex problems. Build a research team where each agent has a role.

## What You'll Build

A multi-agent system where a planner decomposes complex questions into sub-tasks, researcher agents search the web in parallel, a critic reviews findings for accuracy, and a synthesizer combines everything into a coherent report. Fully traced with Langfuse.

## What You'll Learn

- Multi-agent orchestration patterns
- Task decomposition and planning
- Parallel agent execution with async
- Inter-agent communication and handoffs
- End-to-end tracing of multi-agent systems
- LangGraph or raw async orchestration

## Tech Stack

- Python 3.11+
- `openai` or `anthropic` SDK
- LangGraph or raw `asyncio`
- `tavily` or `serpapi` for web search
- Langfuse for tracing
- `structlog` for logging

## Requirements

- Implement 4 agent roles:
  - **Planner** — takes a complex question and breaks it into 3-5 sub-questions
  - **Researcher** (multiple instances) — takes a sub-question, searches the web, and returns findings
  - **Critic** — reviews all findings for consistency, flags contradictions, identifies gaps
  - **Synthesizer** — combines all reviewed findings into a final report
- Researchers run in parallel (async)
- The critic can request additional research if findings are insufficient
- The final report includes:
  - Executive summary
  - Key findings per sub-question
  - Sources cited
  - Confidence assessment
- Integrate Langfuse tracing for the full pipeline (each agent is a span)
- Support configurable number of researcher agents
- Handle failures gracefully (if one researcher fails, the system continues with others)
- Test with at least 5 complex research questions

## Stretch Goals

- Add a "debate" step where two agents argue different perspectives before synthesis
- Implement a feedback loop where the planner can re-plan based on initial findings
- Build a web UI that shows the agent workflow in real-time (which agent is active, what it's doing)

## Hints

- LangGraph's state machine model maps well to this — each agent is a node, transitions are edges
- If not using LangGraph, `asyncio.gather()` with proper error handling works fine for parallel researchers
- The planner prompt is critical — good sub-questions lead to good research. Spend time on this prompt.

## Cost Estimate

~$5-8 per complex research query (multiple LLM calls + search APIs). Use cheaper models for researchers, premium for the critic and synthesizer.

---

[← Back to Agents](../README.md)
