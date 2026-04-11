# Lesson 10 — Structured Outputs and Tool Use

> **The single sentence version:** You almost never want raw text back from a production LLM — you want data in a schema you can parse — and modern APIs give you several mechanisms to force the model's output to fit your schema, ranging from prompting tricks to grammar-constrained decoding that literally cannot produce invalid JSON.

When you're learning LLMs, you mostly work with "chat" — prompt goes in, English text comes out. When you're *building* with LLMs, that stops being useful almost immediately. Your app doesn't want "Sure! Here's a summary of the three best options..." — it wants `[{"name": "...", "price": 4.99}, ...]`. This chapter is about how to reliably get data out of an LLM.

Tool use is the flip side of the same coin: instead of the model writing data into your schema, it's writing a function call that your code will execute. Mechanically, they're the same thing: constraining the model's output to a JSON schema.

---

## The naive approach (and why it fails)

The simplest way to get structured data from an LLM is to ask nicely in the prompt:

> Return a JSON object with fields `name` (string), `age` (integer), `email` (string).

This works... sometimes. In the early days of ChatGPT (2022-2023), this was the only option, and you'd end up with:

- Most of the time: correct JSON
- 3-5% of the time: JSON wrapped in markdown code fences (```json ... ```)
- 1-2% of the time: JSON with trailing commentary ("Here's the JSON you requested:\n\n{...}\n\nLet me know if you need anything else!")
- Sometimes: hallucinated fields not in your schema
- Sometimes: JSON with malformed syntax — unquoted keys, trailing commas, single quotes

In production, "fails 2% of the time" compounds to "fails every minute for one user somewhere." You'd write elaborate parsers that stripped code fences, found the first `{`, balanced braces, fell back to regex, prayed. Everyone who worked with LLMs in that era has a horror story about this.

---

## JSON mode (the first real improvement)

In 2023, OpenAI introduced **JSON mode**: set `response_format={"type": "json_object"}` and the model is guaranteed to output syntactically valid JSON. No more trailing commentary, no more code fences, no more malformed keys.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[...]
)
```

This was a huge relief. But note the careful word: *syntactically* valid JSON. It doesn't guarantee the JSON matches your schema — just that it parses. The model could return `{"wrong_field": "oops"}` and JSON mode would be happy. You still had to validate with Pydantic or JSON Schema and retry on mismatch.

---

## Structured outputs (the current answer)

In August 2024, OpenAI shipped **structured outputs** — a mechanism that guarantees the output conforms to a JSON schema you provide, not just that it's valid JSON. Anthropic and other providers have similar mechanisms now. This is the standard for production use in 2026.

How it works, conceptually: the provider converts your JSON schema into a **context-free grammar**, and at every sampling step, they mask out any token that would lead to a syntactically invalid continuation under that grammar. The model literally cannot produce a token that would break your schema. This is the same technique used by llama.cpp's grammar constraints and open-source tools like `jsonformer` and `outlines`.

### OpenAI structured outputs

```python
from pydantic import BaseModel

class UserRecord(BaseModel):
    name: str
    age: int
    email: str

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract the user info from the text."},
        {"role": "user", "content": "My name is Han, I'm 32, and you can reach me at han@example.com"}
    ],
    response_format=UserRecord,
)

user = response.choices[0].message.parsed   # ← already a Pydantic object, fully typed
print(user.name, user.age, user.email)
```

That's it. No retry loop, no try/except around JSON parsing, no validation error handling (for schema adherence). The model *cannot* output a response that fails to parse into `UserRecord`.

You can also use the lower-level form with a raw JSON schema:

```python
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "UserRecord",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                },
                "required": ["name", "age", "email"],
                "additionalProperties": False,
            },
        },
    },
)
```

**Key requirements of strict mode:**

- `additionalProperties: false` at every object level (no fields outside the schema)
- Every property must be listed in `required` (no optional fields — use unions with null instead)
- Limited schema features: no oneOf at the top level, no `$ref` to external schemas, limited string formats

**Gotchas:**

- The first request with a new schema is slow, because the provider has to compile the grammar (subsequent requests are cached).
- You can still hit *logical* errors the grammar can't prevent. If you ask for a list of exactly 5 items, the schema will accept lists of any length; you need to prompt for the count separately or use schema features like `minItems`/`maxItems`.
- Refusals are returned separately. If the model refuses for safety reasons, you get a `refusal` field instead of a parsed object. Handle this branch.

### Anthropic's approach

Claude doesn't have an exact equivalent of `strict: true` structured outputs as of early 2026. Instead, Anthropic recommends using **tool use with a forced tool choice** — you define a "tool" whose schema is your desired output, force the model to call it, and read the tool call arguments as your structured result:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=[{
        "name": "record_user",
        "description": "Records a user extracted from text",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "age", "email"],
        },
    }],
    tool_choice={"type": "tool", "name": "record_user"},
    messages=[...]
)
user_data = response.content[0].input
```

This works well because Claude's tool calling is schema-constrained under the hood. The end result is effectively the same — schema-valid structured output — using the tool-use surface instead of a dedicated response format.

---

## Tool use (a.k.a. function calling)

Tool use is the same mechanism applied differently. Instead of "give me structured output in this shape," it's "choose which tool to call and give me its arguments in the shape of that tool's schema."

The flow:

1. You define a list of tools, each with a name, description, and JSON schema for its arguments.
2. You call the model with your prompt and the tool list.
3. The model decides:
   - Respond directly with text, OR
   - Call one of the tools — returning the tool name and its arguments.
4. If the model called a tool, *you* execute it (query a database, hit a weather API, run a calculator, whatever).
5. You send the tool result back to the model in the next turn.
6. The model incorporates the tool result and produces a final response (or calls another tool).

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Singapore?"}]
)
```

The model returns a `tool_use` block:

```python
{
    "type": "tool_use",
    "id": "toolu_01ABC...",
    "name": "get_weather",
    "input": {"location": "Singapore", "unit": "celsius"}
}
```

You execute the tool, get `"32°C, humid, rain expected"`, and send it back:

```python
response2 = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather in Singapore?"},
        {"role": "assistant", "content": [tool_use_block]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_01ABC...", "content": "32°C, humid, rain expected"}
        ]}
    ]
)
# The model now produces a text response using the tool result
```

OpenAI has essentially the same API with slightly different field names. Every major provider supports this pattern now.

### Designing good tools

This is the skill. A well-designed tool:

- **Has a name that reads like a function.** `search_documentation`, not `docSearch` or `doSearchMaybe`.
- **Has a description the model will read.** The model decides *when* to call a tool based on its description. Write descriptions in the second person, describe what the tool does, when to use it, and what it returns. Treat it like a tiny system prompt for the tool.
- **Takes a minimal, well-typed argument schema.** Use enums where there are fixed choices. Use descriptions for every field. Fewer, clearer fields beat a deeply nested mess.
- **Returns structured, predictable data.** The result is just text back to the model, so format it clearly. JSON is fine. Well-formatted plain text is fine. A 5000-line stdout dump is not fine — the model will waste context reading it.
- **Fails gracefully.** Don't raise exceptions into the conversation. Return `{"error": "Location not found"}` as a normal tool result and let the model handle it.

Module 4 (Advanced Agents) goes much deeper on tool design. For now, know that tool use is the foundation of every agentic system you'll build.

---

## Parallel tool use

Modern models can call *multiple* tools in one turn — returning a list of tool calls that your code should execute in parallel. If a user asks "What's the weather in Singapore, Tokyo, and Paris?", a well-tuned model will produce three simultaneous `get_weather` calls instead of three sequential turns. This is dramatically faster.

```python
# Response may contain multiple tool_use blocks
for block in response.content:
    if block.type == "tool_use":
        schedule_tool_execution(block)   # run in parallel
```

Both OpenAI and Anthropic support parallel tool calls by default on their current models. You just have to remember to execute them concurrently and return all results together in the next turn.

---

## Streaming structured outputs

You can stream structured output just like regular text. The response arrives field-by-field as the model generates it:

```
{
{"name":
{"name": "Han"
{"name": "Han",
{"name": "Han", "age":
{"name": "Han", "age": 32
...
```

This matters for UX when you want to show partial results as they come in — a chat UI that builds up a form in real time, or a structured report that renders section-by-section. Use the `stream_parse` helper in the OpenAI SDK or the equivalent on Anthropic to get typed partial objects as they arrive.

---

## Common pitfalls

- **Forgetting `additionalProperties: false`.** Strict mode on OpenAI requires it at every object level. Without it, validation fails.
- **Using optional fields when the schema requires all fields listed.** Model the "optional-ness" with a union: `{"type": ["string", "null"]}`. Then handle null in your code.
- **Designing tools the model will never call.** If you give the model a tool but don't describe when to use it, the model may just ignore it and respond with text. Good tool descriptions drive good tool use.
- **Passing raw tool output back without trimming.** A tool that returns 10,000 tokens of log output will eat your context and confuse the model. Summarize or truncate before returning tool results.
- **Assuming schema adherence means semantic correctness.** The model can give you a perfectly-shaped JSON with nonsense values. Schema constraints don't constrain *truth*. You still need to validate business logic.
- **Skipping structured outputs because "it's just one call."** Even for one-off scripts, the reliability win from `strict: true` pays for itself the first time it doesn't crash at 2 AM on an edge case.

---

## What to remember from this lesson

- For any production use, don't parse text output with regex. Use structured outputs or tool calling.
- `strict: true` (OpenAI) and forced tool use (Anthropic) guarantee schema adherence — the model *cannot* produce invalid output.
- Pydantic + `client.beta.chat.completions.parse(...)` is the cleanest Python interface on OpenAI.
- Tool use is the same mechanism applied to "choose and call a function" rather than "fill in a data shape."
- Write tool descriptions like tiny system prompts: the model decides when to call based on them.
- Modern models can call multiple tools in parallel in one turn — use this.
- Schema-constrained decoding guarantees *shape*, not *correctness* — still validate business logic.

Tool use is the bridge from Module 1 (LLMs) to Module 4 (Agents). When you see an agent in Module 4, the core mechanism is the tool-use loop you learned here.

---

## References

- OpenAI, *Introducing Structured Outputs in the API*. https://openai.com/index/introducing-structured-outputs-in-the-api/
- OpenAI, *Structured outputs guide*. https://platform.openai.com/docs/guides/structured-outputs
- OpenAI Cookbook, *Structured Outputs intro*. https://cookbook.openai.com/examples/structured_outputs_intro
- Simon Willison, *OpenAI structured outputs*. https://simonwillison.net/2024/Aug/6/openai-structured-outputs/
- Anthropic, *Tool use with Claude*. https://docs.claude.com/en/docs/build-with-claude/tool-use
- Outlines, *Grammar-constrained decoding for LLMs*. https://github.com/outlines-dev/outlines
- llama.cpp, *Grammars*. https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

---

[← Lesson 9](09-reasoning-models.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 11: Token Economics](11-token-economics.md)
