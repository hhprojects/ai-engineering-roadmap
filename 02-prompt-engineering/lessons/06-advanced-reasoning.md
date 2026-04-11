# Lesson 6 — Advanced Reasoning Techniques

> **The single sentence version:** When chain-of-thought isn't enough, you have a handful of more elaborate techniques — tree-of-thoughts, least-to-most decomposition, ReAct, self-reflection, prompt chaining — each solving a specific weakness of plain CoT at the cost of more compute and complexity.

Chain-of-thought is the default reasoning technique. For most tasks, it's enough. But some tasks break CoT in predictable ways: they branch, they require external knowledge, they benefit from second-guessing. This chapter covers the techniques you reach for when CoT isn't doing the job.

A warning: every technique here is more expensive than plain CoT and more complex to build. Don't reach for them until you've proven that CoT alone isn't working. The mistake is to adopt the fanciest technique you've heard of. The craft is to adopt the *simplest* technique that makes your eval numbers move.

---

## The map of reasoning techniques

To set expectations, here's the lineage:

```
Plain prompt               (zero-shot)
     │
     ▼
Chain-of-thought           (reasoning in one pass)
     │
     ├── Self-consistency  (CoT × N, then vote)
     │
     ├── Tree of Thoughts  (CoT with branching and backtracking)
     │
     ├── Least-to-most     (decompose → solve each piece → combine)
     │
     ├── ReAct             (CoT interleaved with tool calls)
     │
     ├── Reflexion         (CoT + self-critique + revision)
     │
     └── Prompt chaining   (split task across multiple calls)
```

We'll cover each in turn. All of them are alternatives to "do it in one shot." They differ in *where* they add the extra compute: more samples, more branches, more steps, more calls.

---

## Tree of Thoughts (ToT)

**The idea:** chain-of-thought is a single linear reasoning path. Tree of Thoughts maintains *multiple* candidate reasoning paths and searches over them, letting the model backtrack when a path seems bad.

### How it works

Breaking from CoT's linear flow, ToT treats reasoning as a tree:

1. At each step, generate several candidate "thoughts" (partial reasoning steps).
2. Evaluate each candidate — either by asking the model to score it ("sure / maybe / impossible") or by checking some external criterion.
3. Keep the top K candidates and branch further from them.
4. When a branch is clearly bad, prune it. When a branch reaches an answer, evaluate the answer.
5. Search the tree using BFS, DFS, or beam search.

The canonical example from the ToT paper is the **Game of 24**: given four numbers, combine them with arithmetic to reach 24. Example: `4, 9, 10, 13`. A human solution is `(10 - 4) × (13 - 9) = 24`.

Plain CoT fails on this surprisingly often because it commits to one arithmetic path and rides it to a wrong answer. ToT tries multiple paths in parallel, prunes bad ones, and explores good ones — much higher success rate.

### When ToT is worth it

- **Exploratory search problems** where there are many paths and most are wrong
- **Game-playing, puzzle-solving, planning** — tasks with branching decisions
- **Constraint satisfaction** where you need to try-and-backtrack
- **Complex proof construction** where you don't know the right sequence of steps up front

### When ToT is overkill

- Almost everything else. ToT is expensive: you're running the model 10-100× more than you would for plain CoT.
- Tasks where CoT plus self-consistency already works. Self-consistency is usually sufficient for math and logic.
- Anything latency-sensitive.

### A minimal ToT implementation sketch

```python
def tree_of_thoughts(problem, max_depth=3, beam_width=3):
    candidates = [""]  # start with empty reasoning
    for depth in range(max_depth):
        next_candidates = []
        for c in candidates:
            # Generate N new thoughts continuing from this candidate
            thoughts = generate_thoughts(problem, c, n=5)
            # Score each and keep the best
            scored = [(t, evaluate(problem, c + t)) for t in thoughts]
            next_candidates.extend(scored)
        # Keep top-K across all branches
        next_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = [c + t for (t, _) in next_candidates[:beam_width]]
    return best_final_answer(candidates)
```

In practice you'd also add termination conditions (when a candidate solves the problem) and smarter evaluators. The full ToT implementations in research papers are a few hundred lines. You'd only build one of these for a problem that really needs it.

**Reality check:** with reasoning models (o-series, Claude with extended thinking), you rarely need to implement ToT yourself. The model's internal reasoning already does something like beam search over thought candidates, trained into the model via RL. A well-prompted reasoning model can get Game of 24 right without any explicit tree search from the caller.

---

## Least-to-Most Prompting

**The idea:** decompose a hard problem into a sequence of sub-problems, easiest first. Solve each sub-problem, and use the answer to feed the next one.

Originally proposed by Zhou et al. (2022), least-to-most (L2M) addresses a limitation of CoT: CoT struggles when the problem is *longer* or *more compositional* than the examples it saw. L2M works around this by explicitly splitting the problem into pieces.

### How it works

1. **Decomposition stage.** First call: ask the model "Break this problem into a list of sub-problems, simplest first."
2. **Solve stage.** For each sub-problem, make a separate call (or append to the same conversation). Solve each in order. The answer to sub-problem `i` is available when solving sub-problem `i+1`.
3. **Final answer.** Combine the sub-answers into the overall solution.

### Example

Problem: *"Amy climbs a slide. It takes her 4 minutes to climb to the top. She slides down in 1 minute. The water park closes in 15 minutes. How many more times can she slide before it closes?"*

**CoT (may fail):**
```
Let me compute the time per slide: 4 + 1 = 5 minutes. 15 / 5 = 3, so she can slide 3 times.
```

This misses that she's *already* done at least one climb — the "how many more times" is relative to now, not from scratch. L2M helps by forcing decomposition:

**Least-to-Most:**
```
Step 1: Break this into sub-problems.
Sub-problems:
  1. How long does one complete slide cycle take?
  2. How many full cycles fit in 15 minutes?
  3. Is there enough time for the last cycle?

Step 2: Solve sub-problem 1.
A complete cycle is 4 + 1 = 5 minutes.

Step 3: Solve sub-problem 2 using step 2's answer.
15 / 5 = 3 cycles.

Step 4: Solve sub-problem 3 using steps 2 and 3's answers.
3 cycles × 5 minutes = 15 minutes exactly, so she has time for 3 more slides.
```

L2M is especially helpful when:
- The problem has multiple distinct phases that build on each other.
- Examples in the training data are shorter/simpler than the query.
- Plain CoT skips or collapses steps and misses the answer.

It's not usually necessary on modern models (Claude Opus 4.6, GPT-5.4) for school-math-level problems, but it becomes valuable on agent planning tasks where you want to separate "what are the steps?" from "execute each step."

---

## ReAct: Reason + Act

**The idea:** interleave reasoning with actions (like tool calls or search queries). The model thinks about what to do, does it, observes the result, thinks again.

ReAct (Yao et al., 2022) is the pattern every modern agent uses, even if not by that name. We cover it more fully in Module 4 (Agents), but the core loop is central to prompt engineering too.

### The Thought → Action → Observation loop

```
Thought 1: I need to find out the population of Singapore in 2024.
Action 1:  search("Singapore population 2024")
Observation 1: Singapore population in 2024 is approximately 6.04 million.

Thought 2: Now I need the population of Malaysia for comparison.
Action 2:  search("Malaysia population 2024")
Observation 2: Malaysia population in 2024 is approximately 34.3 million.

Thought 3: I have both numbers. Singapore has 6.04M, Malaysia has 34.3M.
           34.3 / 6.04 ≈ 5.68. Malaysia has about 5.68 times Singapore's population.
Action 3:  finish("Malaysia's population is about 5.7x Singapore's, at 34.3M vs 6.04M.")
```

Each iteration:
1. **Thought**: the model reasons about what to do next.
2. **Action**: the model emits a tool call (or a special "finish" action).
3. **Observation**: your code executes the tool, returns the result, and the model sees it.

ReAct is the bridge between prompt engineering and agents. The prompt engineering part is designing the system prompt that teaches the model to produce the `Thought / Action / Observation` format reliably. The agent part is the harness that parses actions, executes them, and feeds observations back.

Modern tool-use APIs (Module 1 Lesson 10) have made explicit ReAct formatting mostly unnecessary: you declare tools via the API's `tools` parameter and the model just calls them, with the Thought/Action/Observation loop happening under the hood in the message format. But the pattern is the same. ReAct is what's happening when Claude Code or any coding agent "does research before editing a file."

### When to use ReAct yourself

If you're using a modern tool-use API: you don't have to implement ReAct explicitly — the API handles it. Just define good tools (Module 1 Lesson 10).

If you're using a model without tool-use API support, or you need to do reasoning over structured sources (a knowledge base, a database, internal documents), implementing the Thought/Action/Observation loop yourself is straightforward and still powerful.

---

## Reflexion and Self-Critique

**The idea:** after producing an answer, the model reviews its own work and revises it.

Reflexion (Shinn et al., 2023) formalized this into a loop where the model:

1. Produces an initial answer.
2. Evaluates the answer (either by running it against a test, comparing to some criterion, or asking itself "is this right?").
3. If the evaluation fails, produces a revised answer using the critique as context.
4. Repeats until the answer passes or you hit a max-iterations limit.

This is the most practical of the advanced techniques for everyday engineering work. You'll use some version of it constantly in production systems.

### Patterns you'll recognize

**Draft → Critique → Rewrite** (the classic form):

```python
draft = model.generate(prompt="Write a persuasive email for ...")

critique = model.generate(
    prompt=f"Here's a draft email. Critique it harshly, pointing out weaknesses:\n\n{draft}"
)

revision = model.generate(
    prompt=f"Rewrite the draft addressing the critique.\n\nDraft: {draft}\n\nCritique: {critique}"
)
```

Three calls. Each one is a focused, simple task. The final output is usually noticeably better than a one-shot draft.

**Code + Test + Fix:**

```python
code = model.generate("Write a Python function that ...")

test_result = run_tests(code)  # actual execution, not model

if test_result.failed:
    fixed = model.generate(
        f"Your code failed with this error: {test_result.error}\n\n"
        f"Original code:\n{code}\n\nFix it."
    )
```

This is how code agents work. The verifier isn't the model — it's real code execution. Closing the loop with a non-model verifier is the key trick that makes Reflexion reliable.

**Extract + Verify:**

```python
extraction = model.generate("Extract the following fields from the document: ...")

verification = model.generate(
    f"Check that every extracted field matches the document:\n\n"
    f"Document: {doc}\n\nExtraction: {extraction}\n\n"
    f"List any fields that are wrong or missing."
)

if verification.has_issues:
    corrected = model.generate(f"Correct the extraction:\n\n{verification}")
```

### When Reflexion is worth it

- The task has **verifiable correctness** (code tests, math checks, schema validation) — then the verifier is reliable.
- The task benefits from **second drafts** in a way that mirrors human writing workflows.
- You have budget for **2-5× the base cost**.
- Latency isn't critical.

### When Reflexion fails

- The critique step is as unreliable as the original answer. If your critique is wrong, you'll "fix" good answers into bad ones. You need a *more reliable* verifier than the generator.
- Creative tasks where there's no objective "right answer." Critique just produces more variance.
- Tasks where the model is confidently wrong in a way it can't self-detect. Self-critique is bounded by the same biases and blind spots as the original answer.

The most robust pattern is: **use an external verifier (code execution, schema validation, another model with a different bias) rather than asking the same model to critique itself.**

---

## Prompt Chaining

**The idea:** split a complex task across multiple LLM calls, passing intermediate results between them. This is the big umbrella that covers Reflexion, Least-to-Most, and most agent-like patterns.

### Why chain at all?

One reason is capacity: a task that doesn't fit in one call (too much output, too much context, too many steps) fits naturally across multiple calls.

Another reason is *inspectability*: if you split "summarize a paper and extract the methodology section" into two calls, you can log and debug each step independently. If you mash it into one call, you're stuck guessing why the extraction went wrong.

A third reason is *cost*: the first step can often be done by a cheap model and the second by a premium model. If only the hard step needs flagship quality, you save money by routing intelligently.

### Common chain patterns

**Extract → Transform → Format.** Pull structured data from text, transform it, render it. Three calls, each focused.

**Research → Write → Edit.** For long-form content, one call to research, one to draft, one to polish.

**Classify → Dispatch → Answer.** Route incoming queries to the right handler based on a classifier step. See also "multi-model router" (Project 3).

**Plan → Execute → Verify.** For complex multi-step tasks: one call to make a plan, subsequent calls to execute each step, a final call to verify the result.

**Map → Reduce.** For batch processing large inputs: split into chunks, process each chunk in parallel (map), combine results (reduce). Canonical for long-document summarization.

### Orchestration

When you start chaining, you move from "LLM call" to "LLM pipeline." You'll need:

- A way to pass structured data between steps (Pydantic models work great)
- Error handling for each step (what happens if step 2 fails?)
- Logging of intermediate results for debugging
- Optional caching of each step so retries don't redo work
- Metrics per step (cost, latency, success rate)

Frameworks like LangChain, LangGraph, and DSPy (Lesson 13) exist specifically to make this easier. You can also roll your own — for small pipelines, a handful of Python functions is often cleaner than importing a framework.

---

## Choosing between techniques

A rough decision tree for when to reach for which technique:

1. **Does plain CoT work on your eval set?** → Use CoT. Done.
2. **Does CoT get close but fail in predictable ways?**
   - Failing on arithmetic / final step wrong? → **Self-consistency** (CoT × 5, vote)
   - Getting stuck on the wrong reasoning path? → **Tree of Thoughts**, or a reasoning model
   - Not breaking the problem into the right pieces? → **Least-to-Most**
3. **Does the task need to look things up or take actions?** → **ReAct** or tool use
4. **Does the task need a draft-and-revise flow?** → **Reflexion** (with a strong verifier)
5. **Does the task naturally split into phases?** → **Prompt chaining**
6. **Is your task really hard and you don't know which of the above to pick?** → Try a **reasoning model** first (Claude with extended thinking, o-series). They internalize many of these techniques and often outperform hand-built chains at comparable cost.

The last point is important in 2026. Many of these techniques were invented to compensate for the reasoning limits of 2022-2023 chat models. A modern reasoning model bypasses much of that work. Before you implement Tree of Thoughts, try Claude Opus 4.6 with extended thinking. It might just work.

---

## Common pitfalls

- **Reaching for advanced techniques before measuring.** If you haven't eval'd plain CoT, you don't know what problem you're trying to fix. Always measure baseline first.
- **Self-critique without an external verifier.** A model critiquing its own work is bounded by its own blind spots. The critique step needs to be genuinely more reliable than the generation step — usually via code execution, schema validation, or a different model.
- **Cost blowup from prompt chains.** A 5-step chain quietly costs 5× what a one-shot call costs. Make sure each extra step is pulling its weight.
- **Reinventing ReAct on top of tool-use APIs.** If your provider's SDK already does tool-use well, use it instead of writing Thought/Action/Observation parsing from scratch.
- **Using Tree of Thoughts when self-consistency is enough.** Self-consistency gets you most of the way for 10× less complexity.
- **Building a reasoning framework before you need one.** DSPy, LangGraph, and similar tools are great once you have a clear pipeline in production. Building with them from day one often slows you down.
- **Applying chaining to creative work.** Splitting poetry into "decompose → solve → combine" doesn't improve the poetry. It just fragments it.

---

## What to remember from this lesson

- Chain-of-thought is the default. Reach for advanced techniques only when CoT provably falls short.
- **Self-consistency** (CoT × N + vote) is the cheapest upgrade and often enough for math/logic.
- **Tree of Thoughts** lets the model search multiple reasoning paths — expensive, powerful on exploration problems, rarely needed on modern reasoning models.
- **Least-to-Most** decomposes hard problems into ordered sub-problems. Useful for long, compositional tasks.
- **ReAct** interleaves reasoning with tool calls — the foundation of every modern agent. Usually handled by the tool-use API automatically.
- **Reflexion** (draft → critique → revise) is the workhorse of production pipelines, especially when the verifier is an external check like code execution.
- **Prompt chaining** is the umbrella: split complex tasks across multiple calls for capacity, inspectability, and cost control.
- In 2026, many of these techniques are made redundant by reasoning models that internalize them. Always try a reasoning model before building elaborate chains.

Next: role prompting and persona control — how to shape tone, voice, and expertise at the system level.

---

## References

- Yao et al. (2023), *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. https://arxiv.org/abs/2305.10601
- Zhou et al. (2022), *Least-to-Most Prompting Enables Complex Reasoning*. https://arxiv.org/abs/2205.10625
- Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models*. https://arxiv.org/abs/2210.03629
- Shinn et al. (2023), *Reflexion: Language Agents with Verbal Reinforcement Learning*. https://arxiv.org/abs/2303.11366
- Madaan et al. (2023), *Self-Refine: Iterative Refinement with Self-Feedback*. https://arxiv.org/abs/2303.17651
- DAIR.AI, *Tree of thoughts*. https://www.promptingguide.ai/techniques/tot
- DAIR.AI, *ReAct prompting*. https://www.promptingguide.ai/techniques/react
- Anthropic, *Chain complex prompts*. https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices

---

[← Lesson 5](05-chain-of-thought.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 7: Role Prompting](07-role-prompting.md)
