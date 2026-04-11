# Lesson 3 — Anatomy of a Good Prompt

> **The single sentence version:** A well-constructed prompt has five ingredients — task, context, examples, format, and tone — arranged so the model reads them top-to-bottom in the order it needs to know them.

You now know the message structure. This chapter is about what actually goes *inside* those messages. We'll break a good prompt into its components, show you the order that tends to work best, and walk through why each ingredient matters. Everything here is general-purpose — it applies to any task, any provider, and any model.

---

## The five ingredients

Almost every effective prompt can be broken into five parts. You don't need all five in every prompt, but knowing the menu helps you recognize when something is missing.

| # | Ingredient | Answers the question | Example |
|---|---|---|---|
| 1 | **Task** | What do you want? | "Summarize this article" |
| 2 | **Context** | What background does the model need? | "The reader is a non-technical executive" |
| 3 | **Examples** | What does a good answer look like? | Input/output pairs showing the pattern |
| 4 | **Format** | How should the output be structured? | "Return JSON with keys `summary` and `tags`" |
| 5 | **Tone / constraints** | What style? What limits? | "Formal, ≤200 words, no jargon" |

The order matters. In long prompts, the model pays more attention to content near the beginning and (to a lesser extent) the end, and less to the middle. A standard ordering that tends to work well:

```
[System]: Persona + durable rules (Lesson 2)

[User]:
  1. Background context      ← what the model needs to know
  2. Task                    ← what you want done
  3. Examples (if any)       ← what "done" looks like
  4. Input / data            ← the actual content to process
  5. Output format           ← restate the expected shape
  6. Final reminder          ← the one thing you really care about
```

This isn't a law — some tasks don't need all of these, and some benefit from a different order. But when you're stuck, fall back to this ordering. It's the "normal form" of a prompt.

---

## Ingredient 1 — Task

The most important part of the prompt. If the task is unclear, nothing else rescues the prompt.

**Bad:**
```
Look at this and tell me what you think.
```

**Good:**
```
Read the product review below and classify it as positive, negative, or mixed.
```

**Better:**
```
Read the product review below and classify it as one of:
  - positive   (the reviewer clearly recommends the product)
  - negative   (the reviewer clearly advises against it)
  - mixed      (genuinely ambivalent or contains both strong positives and negatives)
  - unclear    (the review does not express enough opinion to classify)
```

Notice the progression. Each version is more specific than the last. The "better" version:

- Names the exact categories
- Defines what each category means
- Adds an `unclear` escape hatch so the model doesn't have to force-fit ambiguous cases

The general principle: **specify the task precisely enough that two different annotators would agree on the answer.** If two humans would disagree on whether a given input is "positive" or "mixed," the model will be inconsistent too. Tightening the definitions tightens the outputs.

### Action verbs matter

Words like "analyze," "consider," "think about" invite the model to produce reflections rather than take action. Words like "write," "list," "classify," "extract," "generate" invite the model to produce concrete outputs. Pick verbs that match what you actually want.

A common failure mode with modern models (especially Claude Opus 4.6, as Anthropic explicitly calls out in their docs):

**Unclear — model will suggest:**
```
Can you suggest some changes to improve this function?
```

**Clear — model will act:**
```
Change this function to improve its performance.
```

If you want the model to *do* something, say so.

---

## Ingredient 2 — Context

The model doesn't know your world. Every piece of context you provide either helps it make better decisions or is wasted tokens. The trick is giving enough to ground the response without drowning the real task.

What counts as useful context:

- **Who the output is for.** "A non-technical executive" vs. "a senior engineer" vs. "a 10-year-old" produces very different writing.
- **Why the task matters.** "This will be published on our company blog" vs. "This is an internal memo" shapes tone and formality.
- **What domain you're in.** "In the context of Singapore tax law" vs. "In the context of US tax law" narrows the model's focus.
- **What constraints exist.** "We can only ship by sea freight" rules out solutions that require air shipping.
- **What you've tried before.** "The user already tried restarting; that didn't work" saves the model from suggesting the obvious.

What *doesn't* count as useful context:

- Your feelings about the task ("This is really important to me")
- Filler pleasantries ("Please take your time")
- Repeating the task statement in three different ways
- Meta-commentary about the model's abilities ("You are very smart")

These don't hurt much, but they cost tokens and add noise. Remove them.

### The reason-why trick

Anthropic's prompt engineering guide has a great example. Consider these two instructions:

**Less effective:**
```
NEVER use ellipses
```

**More effective:**
```
Your response will be read aloud by a text-to-speech engine, so never use ellipses
since the text-to-speech engine will not know how to pronounce them.
```

The second version gives the model enough context to generalize. Now the model won't just avoid ellipses — it'll also avoid other punctuation that might confuse TTS, or make other decisions that benefit spoken output. By explaining *why*, you get the rule plus everything the rule implies.

**Rule of thumb:** every time you write a constraint, try to follow it with "because..." and see if adding that reason changes what the model produces. Often it does.

---

## Ingredient 3 — Examples (few-shot)

Showing is more powerful than telling. If you can give the model 2-5 examples of the exact pattern you want, it will imitate the pattern remarkably well — often better than it would from any amount of description.

We cover few-shot prompting in full detail in Lesson 4. For now, the key ideas:

- **Examples are demonstrations, not data.** The model learns the *shape* and *style* of your expected output, not the specific facts.
- **Wrap them in explicit markers.** Anthropic recommends `<example>` tags; other people use `Example 1:` / `Example 2:` headers. Either works as long as the model can tell where examples start and stop.
- **Pick diverse examples.** Cover different inputs and edge cases. If all your examples are easy, the model will only learn the easy pattern.
- **Don't use too few or too many.** 3-5 examples is the sweet spot for most tasks. One example is often not enough; ten is usually overkill and costs tokens.

A quick demonstration:

```
Classify the sentiment of each review as positive, negative, or mixed.

<example>
  <review>This bottle is amazing. Keeps water cold for 24 hours. Worth every dollar.</review>
  <label>positive</label>
</example>

<example>
  <review>Cap broke after a week. Disappointed.</review>
  <label>negative</label>
</example>

<example>
  <review>The bottle is beautiful and the water stays cold, but the handle feels cheap and the price is steep.</review>
  <label>mixed</label>
</example>

Now classify this review:
<review>{user_input}</review>
<label>
```

Three examples, three categories, clearly structured. The model will imitate the pattern even on inputs it's never seen.

---

## Ingredient 4 — Format

Unless you say otherwise, the model will respond in freeform markdown with whatever length it feels like. For production use, you almost always want something more constrained.

Three levels of format control, from weakest to strongest:

**Level 1 — Describe the format in English.**
```
Respond with a one-sentence summary followed by three bullet points of key facts.
```
Works for conversational tasks. Low overhead. No guarantees.

**Level 2 — Provide a template or schema.**
```
Respond using this exact format:

Summary: <one sentence>
Key facts:
  - <fact 1>
  - <fact 2>
  - <fact 3>
```
Better. The model sees the exact structure and imitates it. Works well for markdown-shaped outputs.

**Level 3 — Structured outputs (schema-constrained decoding).**
```python
class ArticleSummary(BaseModel):
    summary: str
    key_facts: list[str] = Field(min_length=3, max_length=3)

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    response_format=ArticleSummary,
    messages=[...]
)
```
Strongest. The model literally cannot produce output that doesn't match the schema. Use this for anything that downstream code has to parse. Module 1 Lesson 10 covers structured outputs in depth; Lesson 8 of this module shows how to combine them with prompt design.

**Rule:** if downstream code consumes the output, use Level 3. If a human reads the output, Level 1 or 2 is usually enough.

### A format-control trick worth knowing

From Anthropic's docs, a surprisingly effective technique: **match your prompt style to your desired output style.** If you want plain prose output, write your prompt in plain prose. If you want markdown-heavy output, write your prompt with markdown headings. The model mimics input style more than you'd expect.

If you want to *reduce* markdown in responses, consider writing your prompt without markdown. Some teams literally remove the `**bold**` from their own system prompts because it was causing the model to overuse bold in its responses.

---

## Ingredient 5 — Tone and constraints

Tone is how the response should feel. Constraints are the hard limits on what it can contain.

Examples of tone instructions:
- "Be warm but concise. Write like a senior engineer explaining to a junior."
- "Formal and precise. No contractions. No casual asides."
- "Conversational and upbeat. Use contractions. Emojis are fine."
- "Detached and neutral. No editorializing. State only what the evidence supports."

Examples of constraints:
- "Response must be ≤200 words."
- "Do not use any words longer than 3 syllables."
- "Never mention our competitors by name."
- "Do not include any URLs in the output."

A few principles:

- **Tell the model what to do, not what not to do.** "Write in flowing prose paragraphs" is stronger than "Don't use bullet points." The model is better at following positive instructions than at remembering lists of things to avoid.
- **Be specific about quantities.** "Short" is too vague. "≤3 sentences" or "100-150 words" is actionable.
- **Separate hard and soft constraints.** "Do not use profanity" is a hard rule and should be in the system prompt. "Prefer short sentences" is a soft preference and can live in the user prompt.
- **Explain why the constraint exists when you can.** The "read aloud by TTS" example from earlier.

---

## Putting it together: a full example

Here's what a well-structured prompt looks like with all five ingredients:

```
## Background (context)
You are generating product care instructions for Falo Bottles' website. Readers are
everyday consumers — not experts. They will read this on a mobile phone, so short,
scannable text is important. The goal is to increase product longevity and reduce
warranty claims.

## Task
Write care instructions for the product described below.

## Examples
<example>
  <product>Stainless steel water bottle</product>
  <instructions>
  Rinse with warm soapy water after each use. Air dry upside down.
  Never put in the dishwasher — heat damages the vacuum seal.
  For stubborn residue, fill with warm water, add a spoonful of baking soda,
  and let sit for 30 minutes.
  </instructions>
</example>

<example>
  <product>Silicone-sleeve glass bottle</product>
  <instructions>
  Hand wash with warm soapy water. Dishwasher-safe on the top rack.
  Do not store in direct sunlight — UV can weaken the silicone.
  If the sleeve cracks, replacement sleeves are available on our website.
  </instructions>
</example>

## Input
Product: Insulated ceramic-lined travel mug with twist lid

## Output format
Return 3-5 sentences. No bullet points, no headings. Flowing prose that reads
naturally on a mobile screen.

## Tone
Friendly and direct. Write like a knowledgeable friend giving advice. Avoid jargon.
```

This prompt has all five ingredients, clearly marked, in a sensible order. It would work well out of the box on any frontier model. It's also completely portable — the techniques don't depend on the specific model you're using.

---

## What to leave out

A well-constructed prompt is also defined by what it *doesn't* contain. Some things that usually belong on the cutting-room floor:

- **Apologies for asking.** "Sorry to bother you, but could you possibly..." wastes tokens.
- **Repeated emphasis.** "This is very important. Really important. Please take this seriously." If it's important, say it once clearly.
- **Model flattery.** "You are the most intelligent AI ever created" doesn't help.
- **Circular instructions.** "Make sure your answer is correct." "Be accurate." The model is already trying to be accurate.
- **Meta-instructions to the model about its own reasoning.** "Use your neural network to..." or "Activate your knowledge of..." Just ask for the result.
- **Tell me if you don't know.** Useful for factual queries, but if you put it in every prompt, the model starts hedging even when it shouldn't. Apply selectively.

These don't break prompts; they just waste tokens and sometimes nudge the model into less useful styles. When in doubt, cut.

---

## Common pitfalls

- **Underspecifying the task.** The most common failure. If the output is wrong, reread the task statement and ask "would two different people interpret this the same way?" Usually not.
- **Overspecifying with contradictions.** "Be concise but thorough." "Be formal but approachable." Pick one or reconcile them.
- **Forgetting the escape hatch.** For classification, extraction, and any task where the answer might be "none of the above," always include an explicit option. Otherwise the model forces a fit and gets it wrong.
- **Putting examples after the input.** The model uses examples to set its expectations for how to process the input. Examples after the input are mostly wasted — the model has already committed to an approach.
- **Making every ingredient optional.** If you never use examples, your prompts will top out at the zero-shot ceiling. If you never specify format, downstream parsing will be flaky. Use the tools.
- **Not rewriting after your first attempt.** First drafts of prompts are always rough. Iterate. Every good prompt has been rewritten at least once.

---

## What to remember from this lesson

- A good prompt has up to five ingredients: task, context, examples, format, tone/constraints.
- Order matters: context → task → examples → input → format reminder.
- Be specific enough about the task that two humans would agree on the answer.
- Provide context the model needs, not context about your feelings. Explain *why* when you give constraints.
- Examples (few-shot) are often more powerful than descriptions. Use `<example>` tags or numbered headers to mark them.
- Tell the model what to do, not what not to do. Positive instructions outperform negative ones.
- Match prompt style to desired output style — it carries over more than you'd expect.
- Cut flattery, repetition, and meta-instructions. They cost tokens and add noise.

Next chapter: zero-shot vs. few-shot, in-context learning, and how many examples to pick.

---

## References

- Anthropic, *Prompting best practices* — examples, XML tags, long context. https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices
- OpenAI, *Prompt engineering guide*. https://platform.openai.com/docs/guides/prompt-engineering
- DAIR.AI, *Basics of prompting*. https://www.promptingguide.ai/introduction/basics
- Lilian Weng, *Prompt engineering*. https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
- Schulhoff et al. (2024), *The Prompt Report*. https://arxiv.org/abs/2406.06608

---

[← Lesson 2](02-message-structure.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 4: Zero-shot, Few-shot, In-Context Learning](04-zero-shot-few-shot.md)
