# Lesson 2 — Message Structure and System Prompts

> **The single sentence version:** Every modern chat API is a list of role-tagged messages — system, user, assistant — and the *system* message is the most important one most people treat as an afterthought.

When you call Claude, GPT, or Gemini, you're not sending a single blob of text. You're sending a structured list of messages, each tagged with a role. This structure is load-bearing: where a piece of instruction lives — system vs. user vs. assistant — changes how the model treats it. This chapter walks through the message format, explains what each role is for, and gives you the intuition to use them deliberately.

---

## The three roles

Every major chat API agrees on this much:

| Role | Who it represents | What it does |
|---|---|---|
| `system` | You, the app developer | Sets the overall behavior, persona, rules, and context |
| `user` | The end user | Whatever the human (or upstream code) is asking right now |
| `assistant` | The model | The model's own previous responses in the conversation |

A minimal request looks like this:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of Japan?"},
]
```

A multi-turn conversation looks like this:

```python
messages = [
    {"role": "system",    "content": "You are a terse, factual assistant. No pleasantries."},
    {"role": "user",      "content": "What is the capital of Japan?"},
    {"role": "assistant", "content": "Tokyo."},
    {"role": "user",      "content": "And its population?"},
]
```

The API appends your new message, runs the model, and returns a new `assistant` message. To keep a conversation going, you append both the user's message *and* the model's response to your running list, and send the whole thing back on the next call. The API is stateless — **the client holds the conversation state, not the server**.

Some older OpenAI APIs also accept a `function` or `tool` role for the results of tool calls; modern APIs fold this into the `tool` or `tool_result` variant. We cover tool messages in Lesson 8 of Module 1 (structured outputs and tool use).

---

## Why the system message matters more than you think

If you've only been writing user-message prompts — typing your instructions into the same place where you'd type a user's question — you're missing most of what prompt engineering can do. The system message is special in three ways:

### 1. It has higher "authority" with the model

Post-training teaches the model that the system message comes from the app developer, not the user, and that its instructions should override or constrain whatever the user asks. This is called the **instruction hierarchy** (OpenAI coined the term; every major provider implements some version of it).

A model that correctly follows the instruction hierarchy will:

- Maintain a persona defined in the system message even when the user tries to change it
- Refuse user-requested actions that violate system-level rules
- Prefer system-level formatting requirements over user-level ones when they conflict

No model is 100% robust to this — prompt injection (Lesson 11) exploits exactly the places where the hierarchy breaks down. But for honest users, the system message is the most reliable place to put durable rules.

### 2. It's cacheable

When you're making many calls with the same system prompt and different user messages — the typical production pattern — the system message can be cached by the provider. You write it once (at a small premium), and every subsequent call reads it from cache at a 90% discount (Module 1, Lesson 11).

**Placement rule:** put everything that doesn't change from call to call in the system message. Put everything that varies per-call in the user message. This arrangement unlocks the most caching benefit.

### 3. It's where persona and rules live durably

A good system message answers, in one place:

- **Who** the assistant is (persona, expertise, tone)
- **What** it's allowed to do (capabilities, refusal conditions, escalation)
- **How** it should format responses (JSON, markdown, specific structure)
- **What context it has** (what domain, what tools, what data it can access)
- **What constraints** apply (length, language, content policy)

You can express all of this in the user message, but then you're repeating it on every turn. The system message is where durable rules live so the conversation history stays clean.

---

## Anatomy of a good system prompt

Here's a pattern that scales well. Adjust for your specific task, but this template is a reasonable starting point:

```
You are <PERSONA>, a <ROLE> that <PRIMARY PURPOSE>.

## Context
<What the user is likely doing, what domain, what tools/data you have access to.>

## Behavior
<How the assistant should respond in general — tone, style, verbosity, level of formality.>

## Rules
<Specific DO and DON'T rules, especially ones that are non-obvious. Explain WHY.>

## Output format
<Exact format expected — JSON schema, markdown structure, length limits.>

## Edge cases
<How to handle ambiguous, off-topic, or harmful requests.>
```

Concrete example — a system prompt for a customer support bot:

```
You are Aria, a customer support assistant for Falo Bottles, a company that sells
reusable water bottles directly to consumers.

## Context
Users are customers asking about their orders, product care, returns, or product
recommendations. You have access to the order_lookup tool (returns order status
by email) and the product_info tool (returns specs for any product in the catalog).
You do NOT have access to payment or account modification tools — escalate those
to a human agent.

## Behavior
Be warm but concise. Prefer short, direct answers (≤3 sentences) unless the user
asks for detail. Use the customer's name if available. Never make up product facts
— always call product_info when asked about specs or materials. If a product
question is outside the catalog, say "That's not one of our products" and offer
to help with something else.

## Rules
- Never promise refunds or returns outside the 30-day window. The policy is strict.
- Never quote a shipping time you didn't get from order_lookup. Shipping times vary.
- Never share order details for an email the user didn't originally provide.
- If the user expresses frustration or asks for a human, offer to escalate
  immediately without defensiveness.

## Output format
Respond in plain conversational text. No markdown, no bullet lists, no headings —
this is read aloud in our app's voice mode.

## Edge cases
- If the user asks about a competitor's product: politely redirect to our catalog.
- If the user asks about an unrelated topic (news, politics, etc.): briefly decline
  and offer to help with Falo Bottles topics.
- If you're uncertain: say so, and offer to escalate. Do not invent answers.
```

Everything above is durable — it applies to every user who talks to Aria. The *user* message just contains the user's current question: "When will order #4821 ship?" That's all that varies per call, so that's all that goes in the user message. Everything else is cached across calls.

---

## Length vs. structure: how much to put in system

A common mistake is either:

- **Too short:** "You are a helpful assistant." The model is now free to do whatever it wants, and different calls will feel wildly different.
- **Too long:** 4000-word essay covering every possible edge case. The model starts losing track, and you're paying for a lot of input tokens on every call.

Guidelines:

- **Start with ~200-500 words.** Enough to set persona, primary rules, output format, and a handful of edge cases. Expand when you see specific failures.
- **One rule per line, not buried in prose.** Bullet points or numbered lists read clearly; paragraphs blur together.
- **Explain the why for non-obvious rules.** "Never use ellipses (this is read by TTS)" beats "Never use ellipses."
- **Use headings (H2/H3 or XML tags) to structure long system prompts.** The model parses headings well and benefits from clear sections.

If your system prompt grows beyond ~2000 words, it's usually a sign that you're trying to handle too many concerns in one prompt. Consider splitting into multiple specialized prompts (routed by the nature of the request) or moving some logic out of the prompt into code.

---

## System vs user: which fields go where

A rule that often trips people up: **per-request context goes in the user message, not the system message.**

Wrong:
```python
system = f"You are a support assistant for order #{order_id}. The customer's name is {name}."
user   = "When will my order ship?"
```

Right:
```python
system = "You are a customer support assistant for Falo Bottles. ..."  # cacheable
user   = f"""
<customer>
  <name>{name}</name>
  <order_id>{order_id}</order_id>
</customer>

<request>
When will my order ship?
</request>
"""
```

In the "wrong" version, every call has a different system prompt, so nothing can be cached. In the "right" version, the system prompt is identical across all users and calls, so the provider caches it and you only pay full price for the small per-call context.

The same logic applies to: retrieved documents (RAG context), session history, timestamps, user-specific preferences. Dynamic → user message. Durable → system message.

---

## Multi-turn conversations: what to keep, what to drop

As a conversation grows, you have to decide what to carry forward. Three strategies, in order of sophistication:

**1. Keep everything.** Append every turn and re-send the whole history. Simple, works until you hit the context window or the bill gets scary.

**2. Keep the last N turns.** Drop anything older. Good for casual chat where long-term context doesn't matter. Risk: the model "forgets" things that mattered if they happened too long ago.

**3. Compact older turns.** Summarize the first N turns into a short recap ("Earlier: user asked X, assistant answered Y; user requested Z which was done"), replace them with the summary, and keep the most recent turns verbatim. This is what production chatbots do. Chapter 9 goes deeper on templates and compaction patterns.

Whichever you pick, two rules are universal:

- **Always keep the original system message verbatim.** Never truncate it.
- **Never drop a message in the middle of a tool-call sequence.** If the assistant made a tool call and received a tool result, those must stay together — dropping one will confuse the model badly.

---

## The quirks of each provider

All major providers use the same three-role structure, but there are small differences you should know about:

### Anthropic (Claude)

- **System message is a top-level parameter, not a role.** You pass it as `system="..."` on the `Messages.create` call, not as a message with `role: "system"`.
- **Alternating turns required.** The `messages` list must alternate `user` and `assistant`. You can't have two `user` messages in a row.
- **First message must be `user`.** Not `assistant`, not `system`.
- **System message can be a string or a list of content blocks** — the list form lets you mark different parts with `cache_control` for fine-grained caching.

### OpenAI (GPT)

- **System message is a role in the messages list**, with `role: "system"`. The first message in the list, by convention.
- **Messages can be in any order**, though conventionally `system → user → assistant → user → assistant → ...`.
- **Supports a `developer` role** on some newer models as an even higher-authority level above `system`. Check the current docs.

### Google (Gemini)

- **Uses `system_instruction` as a top-level parameter** (like Anthropic, not like OpenAI).
- **Messages use `role: "user"` and `role: "model"`** — note "model," not "assistant."
- **Content is a list of `parts`**, which can be text, image, or other media. More structured than text-only content.

If you're writing code that targets multiple providers, abstract the message-building behind a helper function that takes a system prompt, a conversation history, and a user message — then each provider's implementation emits the right shape. Libraries like `litellm` do this for you if you don't want to maintain the abstraction yourself.

---

## The prefill pattern (and its deprecation)

Older Anthropic models supported a neat trick: you could include an assistant message at the *end* of your messages list, partially filled in, and the model would continue from there. It was called **prefilling**.

Example:
```python
messages = [
    {"role": "user", "content": "Extract the name and age as JSON."},
    {"role": "assistant", "content": '{"name": "'},   # prefill
]
```

The model would continue from `{"name": "` and produce something like `Han", "age": 32}`. It was the poor man's structured outputs — you forced the shape by starting the response.

**As of Claude 4.6, prefilling on the final assistant message is deprecated.** Structured outputs (Lesson 8 below, and Module 1 Lesson 10) do the same job more reliably. You'll still see prefill in older code and on non-Claude providers, and Anthropic still supports it on Claude 4.5 and earlier, but don't reach for it in new code. Use structured outputs instead.

---

## Common pitfalls

- **Leaving the system message empty.** If you don't set one, you get whatever default persona the provider configured. For Claude, that's a reasonable general-purpose assistant. For GPT, similar. But you're leaving quality on the table by not taking control of the default.
- **Putting per-request context in the system message.** Breaks caching and bloats your bill. Always ask: "is this the same on every call?" before adding something to the system prompt.
- **Confusing `system` with `assistant`.** A common bug: developers put instructions in an `assistant` message thinking it's "the assistant's rules for itself." That's not what `assistant` means — it's the model's prior *response*. Instructions go in `system`.
- **Assuming the instruction hierarchy is airtight.** It's not. Hostile users can and do override system prompts via prompt injection. Never rely on the system prompt alone to enforce security-sensitive rules. Lesson 11 has the details.
- **Writing the system prompt once and never revisiting it.** System prompts are where technical debt hides. They accumulate clauses as bugs come in, contradict themselves, and bloat. Re-read and prune yours quarterly.
- **Hardcoding the user's question into the system prompt for "testing."** Then forgetting to move it to the user message before shipping. Every senior AI engineer has done this at least once.

---

## What to remember from this lesson

- Every modern chat API takes a structured list of messages tagged with roles: `system`, `user`, `assistant`.
- The `system` message is the highest-authority, most-cacheable place to put durable rules and persona.
- Per-call context (user name, order ID, retrieved documents) goes in the `user` message, not the system message. This is what enables prompt caching.
- A good system prompt has persona, context, behavior, rules, output format, edge cases — in that order, as structured sections.
- Anthropic uses `system=...` as a top-level parameter; OpenAI uses a `system` role in the messages list; Gemini uses `system_instruction`. Same concept, different API shape.
- In multi-turn conversations, always keep the system message and never split tool-call sequences.
- Prefilling used to be how you forced Claude into a specific output shape; use structured outputs instead on Claude 4.6+.

Next chapter: the anatomy of a good user-message prompt — the task, context, examples, format, and tone ingredients that determine whether a prompt works or flails.

---

## References

- Anthropic, *Prompting best practices* (system prompts, XML tags, persona). https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices
- Anthropic, *Messages API reference*. https://docs.claude.com/en/api/messages
- OpenAI, *Chat completions API reference*. https://platform.openai.com/docs/api-reference/chat
- OpenAI, *Instruction hierarchy* (Wallace et al., 2024). https://arxiv.org/abs/2404.13208
- Google, *Gemini API — system instructions*. https://ai.google.dev/gemini-api/docs/system-instructions
- LiteLLM, *Unified interface for multiple LLM providers*. https://docs.litellm.ai/

---

[← Lesson 1](01-what-is-prompt-engineering.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 3: Anatomy of a Good Prompt](03-anatomy-of-a-good-prompt.md)
