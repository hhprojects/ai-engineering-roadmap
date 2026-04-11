# Lesson 9 — Prompt Templates and Caching Patterns

> **The single sentence version:** Once you're making many LLM calls, the difference between amateur and production prompt work is templating — treating prompts like code, versioning them, composing them from parts, and structuring them so the provider's cache can pay for most of your bill.

You can get a long way writing one-off prompts as f-strings. But the moment you have more than 10 calls in flight across more than 2 prompts, you need a discipline for managing them. This chapter is about that discipline — templates, reusable components, versioning, and prompt caching (Module 1 Lesson 11) as a system, not a one-off optimization.

---

## Why templating matters

Imagine you're building a customer support bot. In v1, your code looks like this:

```python
def answer_question(user_name, order_id, question):
    prompt = f"""You are Aria, a support assistant. The customer {user_name}
    has order {order_id}. Answer their question: {question}"""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
```

Two months later, you've added 15 variations of this prompt for different support scenarios. Each one has the same persona, the same rules, the same formatting instructions — slightly different. None of them are cacheable (the customer name is baked into the system prompt). A tweak to the brand voice means editing 15 places. Testing a prompt change means running it against production and hoping.

Every team that builds with LLMs hits this wall. The solution is not "be more disciplined by hand." The solution is templating.

---

## The anatomy of a good template

A template is a parameterized prompt. The *shape* is fixed, the *variables* are filled in at call time. Templates should have:

1. **A clear separation between static and dynamic parts.** Static parts go in the system prompt (cacheable). Dynamic parts go in the user prompt (per-call).
2. **Named parameters, not positional.** Use Python's f-strings or a templating engine (Jinja2) with named placeholders.
3. **A single source of truth.** One file per template, imported everywhere it's used.
4. **Version tags.** Each time you materially change a template, bump its version.
5. **Required inputs and types.** The template module should refuse to render with missing or wrong-typed inputs.
6. **A self-test.** Every template should come with an eval — a small set of inputs and expected outcomes — that runs in CI.

### Level 1: string templating with Python

The simplest form. A function that returns a message list:

```python
SYSTEM_PROMPT = """\
You are Aria, the customer support assistant for Falo Bottles.

## Behavior
Be warm, concise, and direct. Prefer short answers (≤3 sentences).

## Rules
- Never promise refunds outside the 30-day window.
- Never share order details without the customer's email.
- Escalate to a human if the customer is frustrated or the question is outside
  our policies.

## Available tools
- order_lookup(email): returns order status
- product_info(sku): returns product specifications

## Output format
Respond in plain conversational text. No markdown, no bullet lists, no headings.
"""

USER_TEMPLATE = """\
<customer>
  <name>{name}</name>
  <email>{email}</email>
</customer>

<question>
{question}
</question>
"""

def build_support_messages(name: str, email: str, question: str, history: list = None) -> list:
    messages = []
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": USER_TEMPLATE.format(name=name, email=email, question=question)
    })
    return messages
```

Two key things:

- `SYSTEM_PROMPT` is a constant — same on every call, cacheable.
- `USER_TEMPLATE` has placeholders, filled at call time with the per-request data.

Then the call site:

```python
messages = build_support_messages(
    name="Han Hua",
    email="han@example.com",
    question="Where's my order #4821?",
)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=SYSTEM_PROMPT,   # separately — enables caching
    messages=messages,
)
```

This is enough for 90% of production use. Don't reach for fancier tools until this starts to hurt.

### Level 2: Jinja2 for complex templates

When your templates have loops, conditionals, or nested sections, Jinja2 is a good upgrade:

```python
from jinja2 import Template

SYSTEM_TEMPLATE = Template("""\
You are {{ persona.name }}, a {{ persona.role }} for {{ company }}.

## Behavior
{{ persona.tone }}

{% if rules %}
## Rules
{% for rule in rules %}
- {{ rule }}
{% endfor %}
{% endif %}

{% if tools %}
## Available tools
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}
{% endif %}
""")

system_prompt = SYSTEM_TEMPLATE.render(
    persona={"name": "Aria", "role": "support assistant", "tone": "Warm, concise."},
    company="Falo Bottles",
    rules=["Never promise refunds outside 30 days.", "Escalate frustrated users."],
    tools=[
        {"name": "order_lookup", "description": "get order status"},
        {"name": "product_info",  "description": "get product specs"},
    ],
)
```

Now you can reuse the same template across personas, tool sets, and rule sets by swapping the inputs. Tools like LangChain and LlamaIndex use something like this pattern under the hood.

**Caution:** Jinja2 templates are code. They can be bugs. Keep them as simple as possible — resist the urge to put business logic inside `{% if %}` blocks. If a template gets complex, move the logic into Python and render a simpler template.

### Level 3: typed templates with Pydantic

For the most robust production setup, wrap the template inputs in a Pydantic model:

```python
from pydantic import BaseModel, Field

class SupportMessageInput(BaseModel):
    customer_name: str
    customer_email: str
    question: str
    history: list[dict] = Field(default_factory=list)

def build_support_messages(data: SupportMessageInput) -> list:
    ...  # as before, but data.customer_name, data.question, etc.
```

The Pydantic model refuses to construct if fields are missing or wrong-typed. You get IDE autocomplete for template inputs. Schema changes are a compile-time problem, not a runtime one. This is worth the overhead once you have more than a few templates.

---

## Structuring prompts for caching

Recall from Module 1 Lesson 11 that prompt caching can cut your bill by up to 90% on repeated prefixes. The Anthropic implementation uses **exact prefix matching** — it can only reuse content that's identical up to the cache breakpoint.

This constrains how you structure your prompts. The rule: **static content first, dynamic content last.**

### The canonical order

```
[1] Persona / system instructions       ← most static (cache this)
[2] Tool definitions                     ← static (cache this)
[3] Few-shot examples                    ← static (cache this)
[4] Large document context (RAG)         ← changes per session but often reused
[5] Conversation history so far          ← grows each turn
[6] Latest user message                  ← always different (no cache)
```

When you make your API call, you tell Anthropic where to put the cache breakpoint:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT,   # persona + rules, never changes
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": KNOWLEDGE_BASE,  # large reference content
            "cache_control": {"type": "ephemeral"},
        },
    ],
    messages=conversation_messages,   # per-call varies
)
```

Two cache breakpoints. The first at the end of `SYSTEM_PROMPT`, the second at the end of `KNOWLEDGE_BASE`. Subsequent requests with the same system + knowledge base hit the cache for both, paying the normal rate only for the `messages` part.

**Break the rule once and the cache is dead.** If your persona template puts a timestamp at the top ("Today is 2026-04-11"), that timestamp changes every day and the cache never hits. Move the timestamp to the user message.

### What to cache

| Content | Good cache target? |
|---|---|
| System prompt (persona, rules) | **Yes — always** |
| Tool definitions | **Yes — always** |
| Few-shot examples | **Yes — static** |
| Product catalogs, docs, policies | **Yes** — as long as they don't change every minute |
| Conversation history | **Yes** — but use automatic caching or add breakpoints periodically |
| Timestamps, user IDs, session info | **No** — goes in user message |
| Retrieved documents from RAG | **Sometimes** — cache if the same docs are reused across many queries in a session |

### OpenAI's automatic caching

OpenAI caches prompts automatically without markers — you don't configure breakpoints. As long as the first ~1024 tokens of your prompt are stable, you get roughly 50% off on those cached tokens. Less aggressive than Anthropic (90%), but zero configuration.

To benefit: structure your prompts so the stable parts come first. The same "static → dynamic" ordering applies.

### Google Gemini's context caching

Gemini supports **explicit context caching** for long-context workloads: you create a cache object containing your document, then reference it in subsequent requests. Used for repeated Q&A against the same long document. Pricing is per-cache-minute; you pay to keep the cache alive and then each query is cheaper.

Details differ by provider, but the high-level pattern is universal: **structure your prompts so static content is on top, and your bill will drop significantly without any quality loss.**

---

## Multi-turn conversations: compaction as templating

Long conversations grow indefinitely. At some point you hit your context window or blow out your budget. Compaction is how you keep conversations manageable without losing relevant history.

The pattern:

```python
def compact_history(messages: list, keep_last_n: int = 6) -> list:
    if len(messages) <= keep_last_n:
        return messages

    # Summarize the oldest messages into a single recap
    old_messages = messages[:-keep_last_n]
    recent_messages = messages[-keep_last_n:]

    summary = summarize(old_messages)   # a separate LLM call
    recap_message = {
        "role": "user",
        "content": f"<conversation_summary>\n{summary}\n</conversation_summary>"
    }

    return [recap_message] + recent_messages
```

The summary is a special "user" message that captures everything that came before. Recent messages are kept verbatim. The total token count stays bounded even as the conversation grows.

**Key rules:**

- **Always keep the system prompt verbatim.** Never drop or summarize it.
- **Keep the most recent 5-10 turns verbatim.** The model needs recent context for immediate coherence.
- **Never split a tool-call sequence.** If the model made a tool call and received a tool result, those two messages must stay together.
- **Summarize rather than truncate** when you can. Dropping old messages loses context. Summarizing loses detail but keeps the gist.

This is exactly how Claude Code, ChatGPT's long-conversation mode, and most production chatbots manage long sessions. You're not doing anything novel — you're doing what real systems do.

---

## Prompt versioning and testing

Prompts are code. Treat them like code.

### Store them in git

Put your templates in files, not in strings in random places:

```
prompts/
├── support/
│   ├── system_v1.txt
│   ├── system_v2.txt
│   ├── escalation_v1.txt
│   └── tests/
│       ├── test_basic_question.json
│       ├── test_refund_request.json
│       └── test_out_of_policy.json
├── extraction/
│   ├── receipt_v1.md
│   └── tests/
│       └── ...
```

Each template file is version-tagged. Each has a tests directory with expected inputs and outcomes. CI runs the tests on every commit to catch regressions.

### Bump versions on breaking changes

"Breaking" means the output shape changes, the behavior changes meaningfully, or the input parameters change. Bumping a version means:

1. Create a new file: `system_v2.txt`.
2. Update the code to call `v2` for new requests.
3. Keep `v1` in git for rollback.
4. Run an eval comparing `v1` and `v2` on your test set to verify `v2` is actually better.

Versioning prompts like APIs prevents the "quick tweak at 4pm Friday breaks production on Monday morning" class of bug. It sounds overkill for one template; it becomes essential when you have ten.

### Eval on every change

Every time you modify a prompt, re-run the eval. We cover writing prompt evals properly in Lesson 12, but the minimum is:

- 20-100 real or representative inputs
- For each, a "correct" output or a rubric for what counts as correct
- A script that runs the new prompt on each input and computes a quality score
- A comparison against the old prompt's score on the same inputs

If the new version isn't measurably better, don't ship it. Feelings aren't evidence.

---

## Prompt composition: building big prompts from small pieces

Large prompts often get easier to reason about when you build them from reusable components.

```python
# Component: persona
PERSONA_ARIA = """\
You are Aria, the customer support assistant for Falo Bottles.
Warm, concise, direct. Always call tools when you need facts.
"""

# Component: formatting rules for voice mode
VOICE_MODE_FORMAT = """\
## Output format
Plain text for text-to-speech. No markdown, no bullet lists, no punctuation
that sounds weird when read aloud (no ellipses, no em-dashes).
"""

# Component: formatting rules for web UI
WEB_FORMAT = """\
## Output format
Short paragraphs. Markdown allowed. Use **bold** for key terms.
"""

# Component: escalation policy
ESCALATION_POLICY = """\
## Escalation
Escalate to a human if the customer is frustrated, asks to speak to a manager,
or the question is outside our policies.
"""

def build_support_system_prompt(surface: str) -> str:
    format_rules = VOICE_MODE_FORMAT if surface == "voice" else WEB_FORMAT
    return "\n".join([
        PERSONA_ARIA,
        format_rules,
        ESCALATION_POLICY,
    ])
```

Now you have one persona, one escalation policy, two format rule sets — and any combination is available by composition. Change the escalation policy in one place and it propagates everywhere.

**Don't over-do this.** Composition is for components that genuinely need to be shared. If a chunk of prompt is only used once, inlining it is clearer. Premature modularization is the same mistake in prompts as it is in code.

---

## Common pitfalls

- **Baking per-call data into the system prompt.** Breaks caching. Always ask "does this change from call to call?" before adding something to `system`.
- **String concatenation instead of templating.** f-strings are fine until you have 10 of them scattered across a codebase. Centralize early.
- **No version control on prompts.** Every team discovers this the hard way. Put prompts in git on day one.
- **No eval loop on prompt changes.** Changing a prompt without re-running its eval is like merging a code change without running the tests.
- **Caching content that actually changes.** Timestamps, request IDs, random tokens, "today's date." These kill the cache. Check your `cache_creation_input_tokens` vs `cache_read_input_tokens` — if reads are zero, you have a cache-busting bug.
- **Summarizing conversation history by dropping messages.** Real compaction uses a summary LLM call to preserve meaning, not a slice operation that loses it.
- **Building a framework when three templates are enough.** Jinja2, Pydantic models, CI eval — these are overhead. Adopt them when you're feeling the pain of not having them, not before.
- **Fighting the template system instead of the problem.** If your template is getting complex because the task is complex, fix the task (simplify, split into multiple calls). Don't cram more logic into the template.

---

## What to remember from this lesson

- Templates are parameterized prompts with named inputs, stored in version control, tested on real inputs.
- Start with plain string interpolation. Upgrade to Jinja2 when templates need loops or conditionals. Wrap inputs in Pydantic models for type safety.
- Structure prompts as **static content first, dynamic content last** so the provider's cache can hit.
- On Anthropic, use `cache_control` breakpoints to cache system prompts, tool definitions, and large document context. 90% discount on cache reads.
- OpenAI caches automatically (50% discount). Google supports explicit context caching for long documents.
- Long conversations need compaction: summarize old turns, keep recent turns verbatim, never drop tool-call sequences.
- Version your prompts like APIs. Every change gets an eval. Every prompt lives in git.
- Compose large prompts from reusable components, but don't over-modularize.

Next chapter: multimodal prompting — how to send images (and other media) to models, and how to prompt them well.

---

## References

- Anthropic, *Prompt caching*. https://docs.claude.com/en/docs/build-with-claude/prompt-caching
- OpenAI, *Prompt caching*. https://platform.openai.com/docs/guides/prompt-caching
- Google, *Gemini context caching*. https://ai.google.dev/gemini-api/docs/caching
- Jinja2, *Template documentation*. https://jinja.palletsprojects.com/
- LangChain, *Prompt templates*. https://python.langchain.com/docs/concepts/prompt_templates/
- Hamel Husain, *Your AI product needs evals*. https://hamel.dev/blog/posts/evals/

---

[← Lesson 8](08-structured-outputs.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 10: Multimodal Prompting](10-multimodal-prompting.md)
