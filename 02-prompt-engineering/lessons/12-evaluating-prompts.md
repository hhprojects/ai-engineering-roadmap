# Lesson 12 — Evaluating and Iterating on Prompts

> **The single sentence version:** A prompt without an evaluation is not a prompt you trust — it's a vibe — and the single highest-leverage habit in prompt engineering is writing small, honest eval sets that let you iterate on evidence instead of gut feel.

This is the most important chapter in the module. Every technique you've learned is useless without a way to tell whether it's working. You can know every advanced prompting pattern and still ship slop if you're tuning prompts by eye. You can know almost *no* patterns and ship reliable systems if you have a tight evaluation loop.

Writing evals is less glamorous than writing prompts. Do it anyway.

---

## Why evals are hard and important

The honest reason evals aren't routine: they're tedious. They require you to write down the inputs you care about, decide what "correct" means for each of them, and build the plumbing to measure against those decisions. It's all the work of writing tests in normal software, except the thing you're testing is fuzzy and the correctness criteria are judgment calls.

The honest reason evals are worth it: every other technique in this module is guesswork without them. *Is few-shot helping?* Run the eval. *Does chain-of-thought cost me quality on simple tasks?* Run the eval. *Did my "quick tweak to the system prompt" break anything?* Run the eval. When you stop running evals, you're back to "it worked when I tried it one time on my laptop," which is a much weaker standard than the one you should hold your own code to.

Teams that ship reliable LLM products all share one trait: they take evals seriously. Teams that ship unreliable products almost always have a prompt they've never systematically tested.

---

## The minimum viable eval

Before we talk about frameworks and LLM-as-judge and all the sophisticated techniques, here's the minimum you need:

```
1. A small set of input examples (20-100)
2. For each input, what "correct" looks like
3. A script that runs your prompt on each input and checks the output
4. A score: pass rate, accuracy, or something task-specific
```

That's it. Twenty inputs, a notebook, a for-loop, a score. You can build this in an hour. Most production systems run on top of eval sets that started exactly this simple.

Let's make it concrete. Say you're building a support bot that classifies incoming queries as "billing," "technical," "general," or "spam":

```python
# eval_set.yaml
examples:
  - input: "Why was I charged twice last month?"
    expected_label: "billing"
  - input: "The app crashes when I open it on Android."
    expected_label: "technical"
  - input: "What time are you open?"
    expected_label: "general"
  - input: "BUY CRYPTO NOW!!!"
    expected_label: "spam"
  - input: "hi"
    expected_label: "general"
  # ... 20+ more
```

```python
# run_eval.py
import yaml
from my_prompt import classify_query   # the function under test

examples = yaml.safe_load(open("eval_set.yaml"))["examples"]

correct = 0
for ex in examples:
    predicted = classify_query(ex["input"])
    if predicted == ex["expected_label"]:
        correct += 1
    else:
        print(f"FAIL: {ex['input']} → expected {ex['expected_label']}, got {predicted}")

accuracy = correct / len(examples)
print(f"Accuracy: {accuracy:.1%}")
```

Run it. Get a number. Change your prompt. Run it again. Compare numbers.

This is *evals*. Everything else in this chapter is about making this loop tighter, more informative, and applicable to tasks where "correct" is fuzzier than label-matching.

---

## Building a good eval set

Twenty examples is enough to get started. But not just any twenty examples — some are worth far more than others.

### Prefer real data

Evals on synthetic examples you made up in 5 minutes will overfit to your intuitions. Evals on real data from actual users or real documents in production hit the quirks you didn't think of. Always prefer real data when you can get it.

Sources for real data:

- **Production logs** (anonymize PII first) — the gold standard
- **Support tickets, customer emails, historical records** — often already labeled
- **Data from a test deployment** — even 10 users will give you better eval data than 100 synthetic examples
- **Public datasets** — if your task has a public benchmark, use it as a *starting point* (but don't stop there)

### Cover the distribution

A good eval set has examples spanning:

- **Easy cases**: the obvious ones. "What's your return policy?" → general. If your prompt fails on these, something is badly wrong.
- **Medium cases**: the normal-ish ones. "The payment went through but my order doesn't show up yet." → billing or technical? The answer is judgment, and you want to pick one and be consistent.
- **Hard cases**: the ambiguous or tricky ones. "Just wanted to say I love you guys!" → general or spam?
- **Adversarial cases**: the things users actually do that break prompts. Typos. Multiple languages mixed. Prompt injection attempts. Emoji-only messages. Empty input.
- **Edge cases**: the ones you know will break naive implementations. Long inputs. Inputs containing markdown. Inputs in the wrong format.

Aim for a rough distribution like: **50% medium, 20% easy, 20% hard, 10% adversarial.** Your prompt should hit ~100% on easy, ~90% on medium, ~70% on hard, and ~50% on adversarial. If you're at 100% everywhere, your eval is too soft. If you're below 70% on medium, the prompt isn't ready.

### Label honestly

When you assign expected outputs, resist the temptation to label for what you *wish* the model would say. Label for what a thoughtful human reviewer would accept as correct. This means:

- For ambiguous cases, pick one answer and write a comment explaining why.
- For cases you can't decide, exclude them from the eval (or label them as "either X or Y is acceptable").
- If your eval requires domain expertise (medical, legal, financial), get a domain expert to label. Don't fake it.

Some eval sets go further and use *rubrics* instead of single correct answers: "A correct response should mention X and Y and not claim Z." Rubric-based eval is more forgiving and more realistic, especially for open-ended tasks.

### Version your eval set

Treat the eval set like code. Version it in git. When you fix a label, that's a commit. When you add examples, that's a commit. When you retire examples (they're no longer representative), note why.

A versioned eval set becomes your institutional memory of what "working correctly" means for your prompt. It's often more valuable than the prompt itself — the prompt you can rewrite; the eval set is what tells you whether the rewrite was actually better.

---

## Types of metrics

Different tasks need different scoring. Some rough categories:

### Exact match

The simplest. Did the model output the exact expected string (or the expected enum value)?

- **Good for**: classification, extraction of discrete fields, math answers
- **Not good for**: free-form text, anything with stylistic variation

### Substring / regex match

Did the model's output contain the expected string? Or match a pattern?

- **Good for**: extraction tasks where the exact value matters but format varies, structured output where you want to check specific fields
- **Not good for**: tasks where the model might say the right thing in many different ways

### Exact JSON / schema match

Did the model output a JSON object with the exact expected structure and values?

- **Good for**: structured output pipelines, tool call validation
- **Not good for**: tasks where the ordering of fields is free

### Semantic similarity

Does the model's output *mean* the same thing as the expected output? Measured via embeddings (cosine similarity between embedding vectors) or a model comparing them.

- **Good for**: summaries, paraphrases, translation
- **Not good for**: tasks where exact wording matters (legal, medical)

### LLM-as-judge

Another LLM reads the model's output alongside the expected output (or a rubric) and scores it. Prompts like "On a scale of 1-5, how well does this response address the user's question?"

- **Good for**: open-ended tasks where exact-match is too strict and semantic similarity is too fuzzy
- **Not good for**: tasks where the judge itself is unreliable (hard domains, adversarial robustness)

### Task-specific verifiers

Run the model's output through a real verifier. For code generation: does it pass the tests? For math: does the answer match after substitution? For SQL: does it produce the correct rows on the database?

- **Good for**: anything with a ground-truth verifier
- **Not good for**: anything without one — which is most natural language tasks

### Human review

The gold standard and the most expensive. A human reads the model's output and rates it. Usually used alongside other metrics to calibrate them.

- **Good for**: the final word on quality
- **Not good for**: running on every change (too slow and expensive)

A good eval usually combines multiple metrics. For a support bot: exact match on classification + LLM-as-judge on tone + human review on a sample.

---

## LLM-as-judge, done right

Using an LLM to score another LLM's output is the most powerful eval technique in the arsenal, and also the most dangerous. Done well, it scales human judgment to thousands of examples. Done poorly, it teaches you wrong lessons that feel rigorous.

### When it works

LLM-as-judge works when:

- The judge is a *stronger* model than the model being evaluated (e.g., Opus judging Haiku's outputs, or GPT-5 judging GPT-5-mini's)
- The judging task is explicitly framed with a clear rubric
- The judge doesn't know which model produced which answer (avoid confirmation bias)
- You've spot-checked the judge against human labels to validate it isn't systematically biased

### When it fails

LLM-as-judge fails when:

- The judge uses the same model as the system being evaluated (same blind spots)
- The rubric is vague ("rate from 1-10 how good this is" — 10 what?)
- The judge is biased toward longer, more verbose answers (most LLM judges are)
- The task is something the judge itself can't reliably do (if the judge can't answer the question either, it can't reliably grade an answer)

### A judge prompt that works

```
You are evaluating whether a customer support answer correctly addresses the
user's question. You will see the user's question, the assistant's answer,
and a rubric.

<question>
{question}
</question>

<answer>
{answer}
</answer>

<rubric>
An answer is correct if and only if ALL of:
1. It directly addresses the question asked (not a related topic)
2. It does not invent facts not provided in the context
3. It does not promise actions the support bot cannot take
4. Its tone is consistent with our brand (warm but concise, no corporate hedging)

If any of these fail, the answer is incorrect.
</rubric>

Score the answer as "correct" or "incorrect". If incorrect, state the specific
rubric items that failed. Your output must be valid JSON matching this schema:
{{"verdict": "correct" | "incorrect", "failed_criteria": [...]}}
```

Notice the structure:

- Strict rubric with numbered criteria
- Binary verdict (correct/incorrect), not a vague score
- Structured output (JSON) so you can parse it
- Asking for the specific failure reason so you can spot patterns

Run this judge on your eval set, log the verdicts, and sanity-check the judge's decisions on 10-20 examples against your own judgment. If the judge is consistently rating things differently than you would, either the rubric is ambiguous or the judge isn't up to the task.

**One more trick:** calibrate your judge by running it against examples you already know are correct and examples you already know are incorrect. If the judge gets 9/10 right on a mixed set, it's probably usable. If it gets 6/10 right, your judge is not reliable and you need a better prompt, a better model, or human labeling.

---

## The iteration loop

Here's how real prompt engineering sessions look:

```
1. Write version 1 of the prompt (or start from a baseline)
2. Run it on the eval set
3. Read the failures carefully. Look for patterns.
4. Form a hypothesis: "The model is failing because ..."
5. Change one thing in the prompt to address that hypothesis
6. Re-run the eval
7. Compare: did the targeted failures get fixed? Did anything else regress?
8. If improved, commit the change and update the baseline.
9. If worse, revert and try a different change.
10. Repeat until the eval plateaus or you run out of budget.
```

The critical discipline: **change one thing at a time, and always compare against the previous version**. If you change three things at once and the eval improves, you don't know which change helped. You might ship a worse prompt because you bundled a regression with a win.

Another critical discipline: **read the failures, don't just count them**. Aggregate metrics tell you "something is wrong." Reading the actual failing examples tells you *what* is wrong and what to change.

### Common failure patterns and fixes

After running enough evals, you'll recognize the common patterns:

| Failure pattern | Likely fix |
|---|---|
| Model ignores one specific instruction | Move that instruction earlier or emphasize it |
| Model hallucinates facts | Ground in context ("only use information from the `<doc>` below") |
| Model fails on long inputs | Add CoT, or split the task into chain steps |
| Model too verbose | Add length constraint with example of desired length |
| Model too terse | Add example showing desired detail level |
| Model inconsistent across runs | Lower temperature, or use self-consistency voting |
| Model produces wrong format | Use structured outputs / strict mode |
| Model refuses legitimate requests | Reduce adversarial-sounding language in the prompt |
| Model gets math wrong | Add CoT or switch to a reasoning model |
| Model biased toward one class | Reorder or rebalance few-shot examples |

Each of these has been solved thousands of times before. Most of your iteration will be recognizing patterns from this table and applying the standard fix.

---

## When to stop iterating

Prompt engineering has diminishing returns. After the first few iterations, each change tends to improve the metric by less than the previous one. Knowing when to stop is part of the craft.

Signs it's time to stop:

- **Metrics have plateaued.** Three iterations with no meaningful improvement. Your prompt is probably near its ceiling with this model.
- **You're starting to overfit to the eval set.** If you're adding examples to the prompt that exactly mirror the eval cases, you're not really improving the prompt — you're cheating.
- **Quality gains don't justify the token cost.** A 2% improvement that doubles the prompt length is usually not worth shipping.
- **The model itself is the bottleneck.** If even a frontier model can't get past a certain score on your task, it's time to ask whether the model is capable of the task at all. Consider: stronger model, reasoning model, or splitting into a chain of simpler subtasks.

When you stop, record the final eval numbers as your baseline. Ship the prompt. Revisit quarterly, or whenever you see production failures that look systematic.

---

## Regression testing: the habit that pays back

Once you have an eval set, you should run it automatically whenever the prompt changes. This is prompt *regression testing* and it's what separates hobby projects from production ones.

The setup:

- Prompts live in version control
- Eval set lives in version control
- CI runs the eval on every PR that touches a prompt
- A PR that regresses the eval score below a threshold fails CI
- Passing the eval is part of the merge criteria

Now you can't accidentally ship a worse prompt. You can still *deliberately* ship one if the regression is acceptable (smaller model, faster response, different trade-off), but the choice is now explicit and logged.

This is also where eval fidelity matters most. If your eval is 90% synthetic and only 10% real data, you'll pass CI on prompts that fail in production. Invest in realistic eval data.

---

## Eval frameworks and tools

You don't have to build all this from scratch. Several tools exist for running LLM evals at scale:

- **OpenAI Evals** (https://github.com/openai/evals) — the canonical framework, supports any provider, built around YAML eval definitions. A good starting point.
- **Anthropic's eval cookbook** — smaller and more tailored to Claude, focuses on classification and open-ended tasks.
- **Braintrust** (https://braintrust.dev) — SaaS platform for eval management, with a nice UI, versioning, and LLM-as-judge built in. Popular in 2026 for production teams.
- **Langfuse** (https://langfuse.com) — observability and evaluation, integrates with most LLM frameworks. More general-purpose than pure eval.
- **promptfoo** (https://promptfoo.dev) — a CLI tool for running eval sets against multiple providers, with nice diffs between prompt versions.
- **DSPy** (Lesson 13) — doesn't just run evals, but uses them to automatically optimize your prompts.
- **Langsmith** (LangChain's eval offering) — integrates tightly with LangChain pipelines.

You'll encounter all of these in practice. For Module 5 (Observability), we'll go deeper on Langfuse and Braintrust specifically. For this chapter, the point is: **any eval framework is better than no framework**. Pick one early and use it.

But don't let framework choice delay you. Five hours with pyyaml + a for loop is better than five weeks searching for the "right" framework.

---

## Common pitfalls

- **No eval at all.** The most common mistake. "It worked when I tried it" is a memory of one run, not a measurement. Write the eval.
- **Eval too small.** Fewer than ~20 examples is indistinguishable from running one example 20 times. You're not measuring; you're getting anecdotes.
- **Eval too synthetic.** Made-up examples miss the quirks of real data. Use production data when you can.
- **Not versioning the eval set.** An eval that changes while you're measuring is not a measurement.
- **Changing multiple things at once.** You lose the ability to attribute improvements.
- **Not reading failures.** Aggregate accuracy says "X%". Reading failures tells you *what to change*.
- **Optimizing to the eval.** If your changes only work on examples that are in the eval, you've overfit. Hold out a fraction of your data to check generalization.
- **Using LLM-as-judge with a rubric vaguer than "is this good?"**. Concrete, binary, structured criteria. Anything else invites the judge to hedge.
- **LLM-as-judge with the same model being evaluated.** Same blind spots. Use a stronger model for judging, or human review for the final word.
- **Ignoring regressions on "minor" prompt changes.** There are no minor prompt changes. Every change should be evaluated.
- **Running eval once and trusting the number forever.** Models update. Provider quirks change. Your data drifts. Re-run the eval periodically.

---

## What to remember from this lesson

- The most valuable habit in prompt engineering is running evals on every change, no exceptions.
- A minimum viable eval is 20-100 examples with expected outputs and a script that scores them. You can build one in an hour.
- Prefer real production data over synthetic examples. Cover easy, medium, hard, and adversarial cases.
- Pick metrics appropriate to your task: exact match, schema match, LLM-as-judge, task-specific verifier, or human review. Combine them.
- LLM-as-judge is powerful but brittle — use a rubric, use structured outputs, use a stronger model, and calibrate against human labels.
- Iterate by changing one thing at a time, reading the failures, and measuring after every change.
- Common failure patterns have known fixes. Learn to recognize them.
- Stop iterating when metrics plateau, when gains don't justify costs, or when you've hit the model's ceiling.
- Regression-test your prompts in CI. Version the eval set alongside the prompt.
- Eval frameworks exist (OpenAI Evals, Braintrust, Langfuse, promptfoo, DSPy) — pick one early.

Next and final: prompt programming frameworks, where instead of writing prompts by hand you let the framework generate and optimize them for you.

---

## References

- Hamel Husain, *Your AI product needs evals*. https://hamel.dev/blog/posts/evals/
- Eugene Yan, *Evaluating LLMs and RAG pipelines*. https://eugeneyan.com/writing/llm-patterns/
- OpenAI, *OpenAI Evals framework*. https://github.com/openai/evals
- Anthropic, *Creating strong empirical evaluations*. https://docs.claude.com/en/docs/test-and-evaluate/develop-tests
- Braintrust, *Eval platform for LLM applications*. https://braintrust.dev/
- Langfuse, *LLM engineering platform with evals*. https://langfuse.com/docs/evaluation/overview
- promptfoo, *CLI for prompt evaluation*. https://promptfoo.dev/
- Zheng et al. (2023), *Judging LLM-as-a-Judge*. https://arxiv.org/abs/2306.05685

---

[← Lesson 11](11-prompt-injection.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 13: Prompt Programming Frameworks](13-prompt-programming-frameworks.md)
