# 6 — Production

**Goal:** Deploy, secure, scale, and operate LLM applications in production.

## Learning Objectives

- Deploy containerized LLM applications to cloud platforms
- Implement security best practices (prompt injection defense, PII filtering, guardrails)
- Build production infrastructure: caching, rate limiting, monitoring, alerting

---

## 📚 Readings & Resources

| Resource | Type | Why |
|----------|------|-----|
| [LiteLLM Docs](https://docs.litellm.ai/) | Docs | Unified API proxy — model routing, fallbacks, load balancing |
| [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | Guide | **Must-read** — security risks specific to LLM apps |
| [Simon Willison: Prompt Injection](https://simonwillison.net/series/prompt-injection/) | Blog Series | The definitive series on prompt injection attacks |
| [Datadog: LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/) | Blog | Practical guardrails implementation guide |
| [Modal.com Tutorial](https://modal.com/docs/guide) | Docs | Serverless deployment — no infra management |
| [Anthropic: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) | Blog | Production patterns for agent tool execution |

---

## 🔨 Projects

| # | Project | Difficulty | Description |
|---|---------|------------|-------------|
| 1 | [Deploy Your RAG App](projects/01-deploy-rag-app.md) | 🟢 Beginner | Containerize and deploy to a cloud platform |
| 2 | [LLM API Gateway with Caching](projects/02-api-gateway-cache.md) | 🟡 Intermediate | Proxy with semantic caching, rate limiting, cost tracking |
| 3 | [Production Agent Service](projects/03-production-agent-service.md) | 🟠 Advanced | Full production stack: queues, streaming, security, monitoring |

---

## Key Concepts

After completing this section, you should understand:

- Container deployment patterns for LLM apps
- Environment-based configuration management
- Prompt injection attacks and defenses
- PII detection and filtering strategies
- Semantic caching for LLM responses
- Rate limiting and cost control mechanisms
- Health checks, monitoring, and alerting for AI services
- The OWASP LLM Top 10 security risks
- When to use serverless vs. always-on deployment

---

[← Observability](../05-observability/) | [Home](../README.md) | [Next → Career](../07-career/)
