# Lesson 4 — Zero-Shot, Few-Shot, and In-Context Learning

> **The single sentence version:** If you can show the model 2-5 good examples of the pattern you want, it will imitate that pattern without any retraining — this is called "in-context learning" and it's the single most practical trick in prompt engineering.

You learned in Lesson 3 that examples are a powerful ingredient. This chapter explains *why* they work, *when* they work, *when they don't*, and how to pick them deliberately. Along the way you'll meet the academic terms — zero-shot, one-shot, few-shot, k-shot, in-context learning — and understand what each actually means.

---

## What "shot" means

"Shot" is a term borrowed from few-shot classification in traditional machine learning, where you asked a classifier to learn a new class from just a handful of labeled examples. In the LLM context it means:

- **Zero-shot**: You give the model the task with *no examples*. Just the instructions and the input.
- **One-shot**: You give the model *one example* of the task before the real input.
- **Few-shot / k-shot**: You give the model *k examples* (usually 2-10) before the real input.

All three use the same model and the same API. The only thing that changes is how many examples you include in the prompt. No weights are updated. No training happens. The model generalizes from the examples in its context window — a phenomenon called **in-context learning**.

```
Zero-shot:
    [system prompt]
    [task description]
    [input]
    → output

Few-shot:
    [system prompt]
    [task description]
    [example 1: input + output]
    [example 2: input + output]
    [example 3: input + output]
    [input]
    → output
```

That's it. Few-shot is just zero-shot with a few demonstrations glued in.

---

## Why does in-context learning work at all?

This is one of the surprising discoveries of the GPT-3 era and something we still don't fully understand mechanistically. Three intuitions that partially explain it:

1. **The model was trained on text that contained patterns.** A news article with a headline followed by a lede followed by body paragraphs. A Stack Overflow question followed by an answer. A definition followed by examples. The model learned not just language but the *templates* text follows. When you show it `Input: X → Output: Y` twice, you're activating whatever part of the network recognizes "this is a template; produce the next element."
2. **The model's attention mechanism literally reads the examples.** Lesson 5 of Module 1 explained how attention lets a token "look at" all previous tokens. When you include examples in the prompt, every token of the final answer can attend to every token of every example. The model isn't learning from the examples in the training sense — it's *using* them directly as a reference.
3. **Pattern matching is what transformers do.** Papers like *Induction Heads and In-Context Learning* have traced specific attention heads that implement a primitive form of "look for a similar pattern earlier in the context and copy-paste from it." Few-shot prompting is, at a mechanical level, activating those pattern-matching heads.

The practical takeaway: you don't need to "teach" the model anything. You just need to show it the pattern clearly and consistently, and it'll pattern-match on the final input.

---

## When to use zero-shot

Zero-shot is the cheapest option (fewer tokens) and the most flexible (no examples to maintain). Reach for it when:

- The task is **common and well-specified** — classification into standard categories, summarization, translation, standard Q&A. Modern models have seen these tasks millions of times during training.
- The task has **a single correct answer** that the model can reason about without seeing the pattern. Math problems, fact extraction, simple structured output.
- You're **prototyping** and don't yet know what your eval set looks like.
- The cost matters and the task accuracy is "good enough."

An example zero-shot prompt that just works:

```
Classify the sentiment of this review as positive, negative, or neutral.

Review: This bottle is amazing. Keeps water cold for 24 hours. Worth every dollar.

Sentiment:
```

GPT-4, Claude Sonnet, or any modern model will correctly output `positive` with no examples needed.

---

## When to use few-shot

Few-shot is worth the extra tokens when:

- **Your task is unusual or domain-specific.** "Classify these medical notes into one of these 14 ICD-10 codes" is not a task the model has seen thousands of times. Examples help a lot.
- **The output format is tricky.** If you want the output in a specific shape — a particular JSON layout, a particular writing style, a particular level of detail — examples communicate that better than descriptions.
- **Zero-shot fails inconsistently.** If the model is mostly right but occasionally produces a wrong format or misclassifies a subtle case, a few examples often fix it.
- **Edge cases matter.** If you have specific corner cases you want handled ("If the review is sarcastic, treat it as negative"), showing an example of a sarcastic review handled correctly is more reliable than trying to describe the rule in English.
- **You want consistency across calls.** Zero-shot can be creative in ways you don't want. Few-shot anchors the style.

An example where few-shot shines:

```
You are extracting action items from meeting notes. An action item is a specific,
assigned, time-bound task. Return items as a JSON array of
{assignee, task, due_date}.

<example>
  <notes>
  Sarah will finish the Q3 report by Friday. Tom is going to check with legal
  about the contract language — no hard deadline but hopefully this week.
  Someone should also update the roadmap but we don't know who yet.
  </notes>
  <output>
  [
    {"assignee": "Sarah", "task": "Finish Q3 report", "due_date": "Friday"},
    {"assignee": "Tom", "task": "Check with legal about contract language",
     "due_date": "this week (soft)"}
  ]
  </output>
  <note>
  "Someone should update the roadmap" was NOT extracted — no assignee.
  </note>
</example>

<example>
  <notes>
  Alice mentioned she'd love to explore the voice feature idea at some point.
  Bob is working on the dashboard redesign — first draft expected next Wednesday.
  </notes>
  <output>
  [
    {"assignee": "Bob", "task": "First draft of dashboard redesign",
     "due_date": "next Wednesday"}
  ]
  </output>
  <note>
  Alice's "would love to explore" is NOT an action item — no specific task,
  no commitment.
  </note>
</example>

Now extract action items from these notes:
<notes>{notes}</notes>
```

This prompt's real lesson isn't "what an action item looks like" — it's **what a non-action-item looks like**. The examples show exactly how to handle the fuzzy cases (vague commitments, unassigned tasks). A zero-shot prompt with the same instructions would extract too many items.

---

## How many examples do you need?

The honest answer is: *test it*. But some rules of thumb:

- **0 examples (zero-shot)** — easy, common tasks. Default to zero-shot first and only add examples if you see specific failures.
- **1 example (one-shot)** — useful to establish the output format when zero-shot produces the right content in the wrong shape. One example is often enough for format alone.
- **3-5 examples (few-shot)** — the sweet spot for most complex tasks. Enough to cover major variations, not so many that you're bloating the prompt.
- **10+ examples** — diminishing returns. Rarely better than 5, often worse, and definitely more expensive. Only use this many when you have evidence from evals that it helps for your specific task.

Research from Min et al. (2022) and the DAIR.AI prompt guide found a counterintuitive result: **the specific examples often matter less than the format and the range they cover**. What matters is:

1. The format is consistent across examples.
2. The label space (for classification) covers all the classes you care about.
3. The inputs are diverse enough that the model isn't learning a superficial shortcut.

You can sometimes get decent few-shot results with random labels attached to real inputs, as long as the format is consistent. This is weird but true — it suggests the model is mostly learning "what kind of task is this and what shape does the output take," not "what's the correct answer for each specific example."

Practical consequence: don't obsess over picking the "perfect" examples. Pick reasonable ones, make sure they cover your label space, and move on.

---

## How to pick good examples

Given that you're going to include some examples, here's how to pick them:

1. **Cover the label space.** For classification with 5 classes, include at least one example per class. For a free-form task with distinct modes (e.g., "the output might be a number, a range, or 'unknown'"), show an example of each mode.
2. **Cover the difficulty range.** One easy example, one medium, one hard. Don't show only easy cases — the model will fail on hard ones you didn't demonstrate.
3. **Show edge cases explicitly.** If you want the model to return "unknown" when it can't answer, include an example where the input is ambiguous and the output is "unknown". Without this, the model will force-fit.
4. **Keep format perfectly consistent.** Same JSON shape every time. Same delimiters. Same capitalization. If even one example has `"label": "Positive"` and another has `"label": "positive"`, the model will sometimes mix them.
5. **Avoid spoilers.** Don't include an example whose input is extremely similar to the current input — the model may just copy the example's output. Pick examples that are representative but *different* from the actual query.

### Example ordering

Another subtle finding from the research: **the order of your examples affects the output**, sometimes significantly. In particular, the model is biased toward producing labels that appeared in the *last* example, because recent tokens have slightly more influence than earlier ones.

If you're running a classification task and you notice the model is biased toward one class, try shuffling your example order. Better still, randomize the order on every call (if determinism isn't critical). This reduces order-effects.

---

## Few-shot on structured outputs

Few-shot is especially effective when you're using structured outputs (Module 1 Lesson 10, and Lesson 8 below). The schema constrains the shape; the examples teach the model what each field should *contain* semantically.

Without examples, your schema might produce:
```json
{
  "summary": "This article discusses things.",
  "tags": ["stuff", "things"],
  "importance": 5
}
```

With 2-3 good examples showing what good summaries, relevant tags, and thoughtful importance scores look like, you'll get:
```json
{
  "summary": "Senegalese grain farmers face 30% yield losses due to irregular rainfall, threatening local food security.",
  "tags": ["agriculture", "climate-change", "food-security", "west-africa"],
  "importance": 8
}
```

Same schema, different quality. The schema ensures it *parses*; the examples ensure it's *useful*.

---

## When few-shot doesn't help (and sometimes hurts)

Few-shot isn't magic. There are situations where it underperforms:

- **Tasks the model is already great at zero-shot.** Summarizing a news article. Translating between major languages. Basic math. Adding examples wastes tokens with no quality gain.
- **Tasks with a lot of variance in the input.** If the examples are too specific to one kind of input, the model may over-fit to that pattern and fail on different inputs. Solution: more diverse examples, or fewer/no examples.
- **Complex reasoning tasks.** Pure few-shot without reasoning often fails on math, logic, or multi-step problems. You need **few-shot chain-of-thought** — examples that show the reasoning, not just the answer. Lesson 5 covers this.
- **Tasks where the model has a prior bias you're trying to overcome.** If the model is consistently producing a certain style and you're trying to move it away, a few examples often aren't enough — the prior is too strong. You may need more examples, or you may need to switch models.

If you're adding examples and quality isn't improving, stop adding examples. More isn't better — more-diverse or differently-ordered sometimes is. And always ask: is this task zero-shot already? If yes, go back to zero-shot.

---

## A real debugging workflow

Here's how a typical prompt-optimization session looks in practice:

1. **Start zero-shot.** Write the task description and run it on your eval set.
2. **Inspect the failures.** Pick 5-10 cases where the output is wrong. Look for patterns.
3. **Decide: is this a task problem or a format problem?**
   - Format problem → add 1-2 examples showing the right format. Re-run.
   - Task problem → add examples that cover the kinds of inputs you're failing on.
4. **Re-run the eval.** Did the failures you targeted get fixed? Did any *new* failures appear on cases that used to work? (This happens more than you'd expect — adding examples can over-fit to the examples and break things that were fine.)
5. **Iterate.** Each cycle should be one change at a time: add one example, swap an example, reorder examples. Measure after each.
6. **Stop when the eval plateaus** or you run out of budget for prompt engineering. Sometimes the answer is "this model can't do this task; try a stronger model."

This loop is the core of practical prompt engineering. Everything else is variations on "add / remove / modify one thing and measure."

---

## Common pitfalls

- **Adding examples without a reason.** If zero-shot works, leave it alone. Examples cost tokens and can degrade performance on tasks they weren't chosen for.
- **Mixing formats between examples.** Inconsistent formatting confuses the model. Pick one shape and stick to it across all examples.
- **Forgetting to include a negative example.** If your task needs an "unknown" or "none" option, show at least one example where the answer is that option. Otherwise the model won't use it.
- **Examples leaking into production.** If your test examples accidentally contain real customer data, you've now baked that data into every API call. Treat prompt examples like code — review what's in them.
- **Using examples from a different distribution than production input.** If your examples are neatly formatted but your real input is messy, the model will learn on clean data and fail on messy data. Match the example style to the real thing.
- **Ignoring example order.** If your model is biased, try shuffling. Then verify the bias is gone.
- **Not updating examples when the task evolves.** Examples are code. They rot. Every time the task definition changes, audit the examples.

---

## What to remember from this lesson

- Zero-shot = no examples. Few-shot = a handful of examples. Same API, same model — just a longer prompt.
- In-context learning is real and practical. Examples genuinely improve quality on non-trivial tasks.
- Start zero-shot; add examples when you see specific failures. Don't add examples preemptively.
- 3-5 examples is the sweet spot. More isn't usually better.
- Pick examples that cover the label space, the difficulty range, and any tricky edge cases. Consistency of *format* matters more than which specific examples you pick.
- Example order affects outputs — watch for recency bias, shuffle if you see it.
- Few-shot shines for unusual tasks, tricky formats, and consistency. It's redundant for tasks the model already does well zero-shot.
- For complex reasoning tasks, combine few-shot with chain-of-thought (next chapter).

Next up: chain-of-thought prompting — the technique that turned "please think step by step" from a joke into one of the most important interventions in prompt engineering.

---

## References

- Brown et al. (2020), *Language Models are Few-Shot Learners* (the GPT-3 paper). https://arxiv.org/abs/2005.14165
- Min et al. (2022), *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*. https://arxiv.org/abs/2202.12837
- Olsson et al. (2022), *In-context Learning and Induction Heads*. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- DAIR.AI, *Few-shot prompting*. https://www.promptingguide.ai/techniques/fewshot
- Anthropic, *Use examples (multishot prompting)*. https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices
- Lu et al. (2022), *Fantastically Ordered Prompts and Where to Find Them* (example ordering). https://arxiv.org/abs/2104.08786

---

[← Lesson 3](03-anatomy-of-a-good-prompt.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 5: Chain-of-Thought](05-chain-of-thought.md)
