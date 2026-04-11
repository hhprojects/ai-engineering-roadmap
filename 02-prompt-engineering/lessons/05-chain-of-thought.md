# Lesson 5 — Chain-of-Thought Prompting

> **The single sentence version:** Asking the model to "think step by step" before answering dramatically improves its accuracy on math, logic, and multi-step reasoning tasks — because it gives the model room to do the reasoning in its output tokens instead of trying to do it all in one shot.

Chain-of-thought (CoT) is the single most famous prompt engineering technique, and for good reason. It's simple, cheap, and effective. It's also more subtle than it looks — knowing *when* CoT helps (and when it doesn't) saves you from wasting tokens and latency on tasks that wouldn't have benefited.

---

## The idea, in one example

Classic example from the Wei et al. (2022) paper:

**Prompt (zero-shot, no CoT):**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3
tennis balls. How many tennis balls does he have now?
A:
```

Early models often produced **11** — wrong. (They somehow decided 5 + 2 × 3 = 11 when ordered left-to-right.)

**Prompt (zero-shot CoT — add one line):**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3
tennis balls. How many tennis balls does he have now?
A: Let's think step by step.
```

Output:
```
Roger starts with 5 tennis balls. 2 cans of tennis balls, each containing 3 balls,
is 2 × 3 = 6 additional balls. So in total Roger has 5 + 6 = 11 tennis balls.
```

Wait, that's still wrong — or is it? Read more carefully: `5 + 6 = 11`. That's correct. The first "wrong" answer was 11 computed *incorrectly* (the model stumbled through the arithmetic); the CoT answer is 11 computed *correctly* (walking through each step explicitly). On harder problems, the difference is night and day.

The 2022 finding that startled the field: simply appending **"Let's think step by step"** to a math prompt raised GPT-3's accuracy on the GSM8K math benchmark from ~18% to ~41%. That one phrase. No retraining, no examples. Just asking the model to do the reasoning out loud.

---

## Why it works

CoT works for two related reasons.

### 1. Models compute incrementally

Remember from Module 1 Lesson 6 that the model generates tokens autoregressively — one at a time, with each new token able to attend to all previous tokens. Every token is a chance to do a little more computation.

When you ask "what's 12,345 × 67,890 in one shot?", the model has to produce all eight digits of the answer in its very first prediction. It can't. The math is too hard for the single forward pass.

When you ask "what's 12,345 × 67,890? Think step by step.", the model writes:
```
12,345 × 67,890
= 12,345 × 67,000 + 12,345 × 890
= 826,515,000 + 10,987,050
= 837,502,050
```

Each line is a separate generation step. Each step is small enough to get right. And each step's result is available for the next step to use. The model is effectively **using its output tokens as a scratchpad**.

You can think of this as giving the model more "thinking time" — but more accurately, you're giving it more *intermediate state* to work with. Every reasoning step that ends up in the output becomes part of the context the next step can attend to.

### 2. It activates patterns from training

Chain-of-thought prompting also nudges the model toward text it's seen before. Step-by-step solutions appear all over the training data — textbook proofs, forum answers, tutorials, Stack Overflow explanations. The phrase "let's think step by step" activates the part of the model's distribution that contains these carefully-reasoned explanations. The model starts behaving like it's writing a math textbook solution, because that's the pattern it pattern-matched to.

This is also why CoT helps *most* on the kinds of problems that appear in training data with step-by-step solutions — math, logic, science, code debugging — and *least* on open-ended creative tasks where there's no canonical step-by-step template to imitate.

---

## The three flavors of CoT

There are three variants of CoT that you'll see referenced. They're all minor variations on the same idea.

### Zero-shot CoT

The simplest. Add a single phrase to the end of your prompt — usually "Let's think step by step" — and let the model produce its reasoning. No examples needed.

```
What is the probability that a random permutation of the letters in MISSISSIPPI
starts with the letter I? Let's think step by step.
```

Pros: costs almost nothing. Works on a surprisingly wide range of tasks.
Cons: you can't control the *format* of the reasoning; the model picks its own structure. Less reliable than few-shot CoT on hard problems.

Variants of the magic phrase that all work similarly well:
- "Let's think step by step"
- "Think through this carefully"
- "Reason about this before answering"
- "Explain your reasoning before giving the final answer"

Modern Anthropic documentation explicitly mentions that Claude Opus 4.5 is particularly sensitive to the word "think" when extended thinking is *disabled* — so variants like "consider," "evaluate," or "reason through" sometimes work slightly better on Claude.

### Few-shot CoT

Like zero-shot CoT, but you also show the model 2-5 examples of problems *with their reasoning spelled out*. This is more reliable on harder problems because it shows the model exactly the style and depth of reasoning you want.

```
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove
   today. After they are done, there will be 21 trees. How many trees did the
   grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were
   planted. So there must have been 21 - 15 = 6 trees planted.
   The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars
   are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
   The answer is 5.

Q: [your actual problem]
A:
```

Notice the format: question → step-by-step reasoning → "The answer is X." The model learns this template and follows it on the new question.

Pros: more reliable, more controllable. You can shape the reasoning style.
Cons: costs more tokens (examples + reasoning), and you have to maintain the examples.

### Automatic CoT (Auto-CoT)

Auto-CoT (Zhang et al., 2022) tries to automate the example generation. Instead of hand-writing CoT examples, you:

1. Cluster your problems by similarity.
2. From each cluster, pick one representative problem.
3. Generate its reasoning using zero-shot CoT.
4. Use those auto-generated reasoning chains as your few-shot examples.

This is clever but rarely needed in practice. Modern models are good enough at zero-shot CoT that the extra plumbing isn't usually worth it. Mention it so you recognize the term.

---

## Separating reasoning from the final answer

A practical problem with CoT: you often want to show *only* the answer to the user, not the reasoning. Letting the model ramble about its calculations makes for a messy UI.

The clean solution: ask the model to wrap its reasoning in one set of tags and its final answer in another.

```
Work through this problem carefully. Put your reasoning inside <thinking> tags and
your final answer inside <answer> tags.

Question: [question]
```

Output:
```xml
<thinking>
The problem involves... Let me break it down. First, ... Next, ...
Therefore, ...
</thinking>
<answer>
42
</answer>
```

Now your code can extract everything inside `<answer>...</answer>` and discard the thinking. The reasoning still happened — it still helped the model get the right answer — but it's invisible to the user.

This trick is also how to combine CoT with structured outputs. You use `<thinking>` tags for reasoning and then define a schema that starts with the final answer. The model reasons inside the thinking block, which doesn't have to match your schema, and then produces a schema-valid answer at the end.

---

## Self-consistency: CoT with voting

Even with CoT, models sometimes make arithmetic mistakes or take a wrong reasoning path. **Self-consistency** (Wang et al., 2022) improves reliability by sampling the model multiple times and taking a majority vote.

The recipe:

1. Ask the same CoT question `N` times (e.g., 5 or 10 times), each with temperature > 0 so you get different reasoning paths.
2. Extract the final answer from each.
3. Take the most common answer as the final result.

```
# Pseudo-code
answers = [
    model(prompt, temperature=0.7)
    for _ in range(5)
]
final_answers = [extract_answer(a) for a in answers]
result = Counter(final_answers).most_common(1)[0][0]
```

Why it works: different reasoning paths are likely to converge on the correct answer more often than they converge on any one specific incorrect answer. You're trading compute (more API calls) for accuracy.

Classic example from the paper: *"When I was 6, my sister was half my age. Now I'm 70 — how old is my sister?"* CoT alone got this wrong sometimes (producing answers like 35 or 70). Self-consistency with 5 samples produced answers [67, 67, 35, 67, 67], and the majority vote picked the correct 67.

**When self-consistency is worth it:**
- Math, logic, or any task with a single correct answer
- Tasks where you'd otherwise need to manually verify outputs
- You have compute budget for 3-10× more calls

**When it isn't:**
- Creative tasks (what does "majority vote" even mean for a haiku?)
- Tasks with a clear verifier — just check the answer with code instead
- Latency-sensitive applications — you're multiplying latency by N

Self-consistency is mostly useful for batch workloads and evaluations. For interactive chat, the latency cost usually isn't worth it.

---

## CoT vs. reasoning models

Here's a subtle point that matters in 2026: **reasoning models do chain-of-thought automatically, for you, in hidden tokens.**

Recall from Module 1 Lesson 9 that models like OpenAI's o-series, Claude with extended thinking, and DeepSeek R1 are trained to produce long internal reasoning *before* their visible output. The model decides how much to think; you don't have to ask it to.

So when you're using a reasoning model, **you usually don't need to add "let's think step by step" to your prompt.** The model is already doing the reasoning. Adding CoT instructions on top can actually waste tokens — you're asking for more visible reasoning on top of the hidden reasoning the model was already going to do.

Rule of thumb:

- **Non-reasoning models (Sonnet, GPT-5, Haiku, Llama)**: use CoT when the task is multi-step. It's the cheapest quality boost in the book.
- **Reasoning models (Opus 4.6 with thinking, o-series, R1)**: don't add CoT. The model is doing it automatically. Focus your prompt on the task description and let the model's built-in reasoning handle the rest.

When in doubt, test both. On some tasks, even reasoning models benefit from being explicitly told to *show their work* in the visible output (as opposed to keeping it hidden in thinking blocks). But the days of "always add 'let's think step by step'" are over for reasoning models.

---

## When CoT helps vs. when it doesn't

A rough guide to when chain-of-thought is worth the extra tokens:

**CoT helps a lot:**
- Math word problems, arithmetic chains, unit conversions
- Logic puzzles, syllogisms, constraint satisfaction
- Multi-step code reasoning ("what does this function return for this input?")
- Scientific reasoning ("given these observations, what's the likely cause?")
- Any task where the answer requires *composing* multiple facts

**CoT helps a little:**
- Classification with complex criteria (the model can reason about which category fits)
- Extraction from long documents (the model can identify relevant passages first)
- Drafting → critiquing → rewriting loops (where each stage is one call in a chain)

**CoT doesn't help (or hurts):**
- Pattern-matching tasks: "is this spam?" "what language is this?" "extract the email address"
- Simple factual lookups the model can answer from memory
- Creative writing — "let's think step by step about this haiku" produces stilted haiku
- Pure translation (reasoning doesn't improve it)
- Tasks already handled well zero-shot — you're paying for reasoning that isn't needed

**Test before committing.** The quality-vs-cost trade-off of CoT depends on the specific task and model. Always measure, and remove CoT from your prompt if it doesn't measurably help.

---

## A practical template

Here's a template that combines everything in this chapter:

```
[Task description]

[Optional: 2-3 few-shot examples showing reasoning + answer in the format below]

Now apply this to the input below.

<input>
{input}
</input>

Work through your reasoning carefully inside <thinking> tags. Consider edge cases
and alternative interpretations. Then put your final answer inside <answer> tags,
matching the format shown in the examples.

<thinking>
```

By prefilling `<thinking>` at the end, you're committing the model to start with reasoning (this works on older Claude models; on Claude 4.6+ where prefill is deprecated, just ask for the format in the instructions and the model will comply without the explicit prefill).

---

## Common pitfalls

- **Using CoT on reasoning models.** If the model already thinks internally, asking for CoT on top is redundant — and on some models, it can actually constrain the reasoning style in ways that hurt quality.
- **Leaving the reasoning in the user-facing response.** If your app shows the raw output to users and you didn't wrap the reasoning in tags, they'll see paragraphs of the model's internal monologue. Always separate reasoning from answer.
- **Trusting one CoT run on a hard math problem.** The model can reason confidently to the wrong answer. For anything where correctness matters, use self-consistency (multiple samples + vote) or verify with code.
- **Adding "step by step" to prompts that don't benefit.** For simple tasks, CoT just makes responses longer and slower without improving quality. Always test.
- **Using few-shot CoT examples with inconsistent reasoning styles.** If one example uses bullet points and another uses prose and a third uses numbered steps, the model won't learn a consistent pattern. Pick one reasoning style and use it across all examples.
- **Forgetting that CoT is tokens.** A prompt with 3 few-shot CoT examples plus a detailed thinking section can easily quadruple your per-call cost. Budget for it.
- **Not showing the *wrong* path when it's instructive.** For tasks with common mistakes, it sometimes helps to include an example like *"Wait — if we applied this rule naively we'd get X, but that's wrong because..."* This teaches the model to check its own work.

---

## What to remember from this lesson

- Chain-of-thought prompting asks the model to reason out loud before committing to an answer. It dramatically improves performance on math, logic, and multi-step reasoning.
- Zero-shot CoT (`"Let's think step by step"`) is the simplest version. Few-shot CoT (examples that show reasoning + answer) is more reliable on hard problems.
- CoT works because the model uses its output tokens as a scratchpad — each reasoning step is computed, written, and then available for the next step to use.
- Wrap reasoning in `<thinking>` tags and the answer in `<answer>` tags to separate them cleanly. This is the standard pattern.
- Self-consistency (sampling N times and voting) further improves reliability on tasks with a single correct answer.
- Reasoning models (o-series, Claude with thinking, R1) do CoT internally — don't add CoT prompts on top of them.
- CoT doesn't help simple tasks, pattern-matching, or creative writing. Measure before committing to it in production.

Next chapter: when CoT isn't enough — more advanced reasoning techniques like Tree of Thoughts, least-to-most decomposition, ReAct, and self-reflection.

---

## References

- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. https://arxiv.org/abs/2201.11903
- Kojima et al. (2022), *Large Language Models are Zero-Shot Reasoners* (the "let's think step by step" paper). https://arxiv.org/abs/2205.11916
- Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning*. https://arxiv.org/abs/2203.11171
- Zhang et al. (2022), *Automatic Chain of Thought Prompting in Large Language Models*. https://arxiv.org/abs/2210.03493
- DAIR.AI, *Chain-of-thought prompting*. https://www.promptingguide.ai/techniques/cot
- DAIR.AI, *Self-consistency*. https://www.promptingguide.ai/techniques/consistency
- Anthropic, *Let Claude think (chain of thought)*. https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices

---

[← Lesson 4](04-zero-shot-few-shot.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 6: Advanced Reasoning](06-advanced-reasoning.md)
