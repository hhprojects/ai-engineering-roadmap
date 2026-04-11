# Lesson 7 — Role Prompting, Personas, and Tone Control

> **The single sentence version:** Assigning the model a role or persona is a cheap, effective way to steer tone, expertise, and default assumptions — one sentence in your system prompt can shift the whole character of the output.

You can be as precise as you like with task instructions and still get responses that feel *wrong* — too formal, too chatty, too hedged, too corporate. Role prompting is the tool for fixing that. It's one of the shortest-to-describe techniques in the whole module and one of the most underrated.

---

## The basic idea

A role prompt is a short instruction — usually in the system message — that tells the model who to *be*.

```
You are a senior software engineer reviewing a junior colleague's code.
```

vs.

```
You are a supportive coding mentor helping a beginner.
```

Same task ("review this code"), wildly different outputs. The first will be terse, direct, focused on maintainability and edge cases. The second will be warm, encouraging, include explanations of basic concepts. No change to the task instructions — just the role.

Anthropic's docs put it this way: even a single sentence in the system prompt changes the model's behavior. A role prompt is the smallest possible intervention for the biggest possible vibes shift.

---

## Why it works

Recall from Lesson 1 that the model was trained on text written *by* lots of different people *in* lots of different roles. When you say "you are a senior software engineer," you're activating the region of the distribution that contains text written by senior software engineers — more technical vocabulary, more direct tone, less hand-holding, different priorities.

The model doesn't "become" a senior engineer in any meaningful sense. It just matches its output style to text that matches the role description. This is pattern completion, same as everything else the model does — but since the training data contains so many distinct writing styles, even short role descriptions can pick out recognizable ones.

---

## What a good role prompt includes

At minimum, a role prompt should answer three questions:

1. **Who are you?** — the persona (job title, level, domain)
2. **Who are you talking to?** — the implied audience
3. **What do you care about?** — the priorities/values that shape your judgments

A minimal example:

```
You are a Singapore-based tax advisor helping a small business owner who has
never filed a company tax return before. You care most about keeping them
compliant while flagging any tax deductions they might not know about.
```

Three sentences. Covers role, audience, priorities. This alone will massively shape the model's tone and the kinds of details it volunteers — compared to just "You are a tax advisor."

---

## Levels of role specificity

Roles exist on a spectrum from "generic" to "named character." Each level has its uses.

### Level 1 — Professional role

```
You are a technical writer.
```

The lightest touch. Steers vocabulary and tone without over-constraining. Good for general-purpose applications where you want a consistent voice but need flexibility.

### Level 2 — Qualified professional role

```
You are a technical writer with 15 years of experience writing API
documentation for large-scale web services. You write in short, scannable
sentences and always include a code example for every concept.
```

More specific. Brings in writing style ("short, scannable") and habits ("code example for every concept"). Good for content generation where you want a recognizable style.

### Level 3 — Named character / brand persona

```
You are Aria, the friendly but precise AI assistant for Falo Bottles. Your
personality is warm, slightly playful, and never corporate. You respond
in short conversational sentences. You never use the phrase "I apologize
for the confusion" because it sounds corporate.
```

Full persona with a name, brand-aligned voice, explicit dos and don'ts. Good for customer-facing products that need a consistent brand voice across every interaction.

### Level 4 — Fictional or historical character

```
Respond in the voice of Sherlock Holmes, the fictional detective.
```

This sometimes works for creative writing, but it's less useful for production applications. Models vary on how well they can stay "in character" without adding unwanted flourishes. Usually a distraction — reach for it only when you genuinely need period-appropriate writing.

The useful levels for production work are 1-3. Level 4 is fun to experiment with but rarely what you ship.

---

## Combining role with task

Role prompts live in the system message. Task instructions live either in the system message (if the task is durable) or the user message (if the task varies per call). They work best together, not instead of each other.

**Bad (role replaces task):**
```
System: You are an expert prompt engineer.
User:   Help me with this.
```

The role is there but the task is empty. The model will produce *something* prompt-engineering-related, but it'll be generic advice.

**Good (role + task):**
```
System: You are a senior prompt engineer. You respond with specific, testable
        recommendations. You explain the reasoning behind each recommendation
        in one sentence. You never give vague advice like "try different phrasings."

User:   Here's my current prompt. It's producing responses that are too long.
        How should I change it?

        <prompt>{prompt}</prompt>
```

Now both the role (who you are, how you think) and the task (what to do right now) are clear. The model has something to act on.

---

## Tone control: adjectives vs. examples

A common mistake: thinking that tone can be fully captured by adjectives.

**Weak:**
```
Your tone should be friendly, professional, concise, warm, helpful, knowledgeable.
```

Six adjectives. None of them really tell the model what you want. "Friendly" means different things in different contexts. "Concise" is famously relative.

**Stronger:**
```
Here are three examples of the tone we want:

<example>
User: My bottle leaks.
Response: Ugh, that's so frustrating — leaky bottles are a mess. Can you tell me
which model you have? If it's the Urban series, there's usually a loose O-ring
you can reseat in about 30 seconds. If it's the Summit series, the fix is different.
</example>

<example>
User: Is the Summit 24oz dishwasher safe?
Response: Short answer: no. The vacuum seal on the Summit series can degrade in the
dishwasher's heat. Hand wash with warm soapy water — takes just a minute.
</example>

<example>
User: Do you have bottles in blue?
Response: We do! The Summit 20oz and Urban 16oz both come in a matte navy blue.
Both are on our site under Colors → Navy.
</example>
```

Three examples. The model will now match the tone of those examples — lightly informal, sympathetic, direct, includes specifics, doesn't ramble. This is dramatically more reliable than adjectives alone.

**The general principle:** when you can show, show. When you must tell, tell with *specifics* and *explanations* rather than vague adjectives.

---

## Negative constraints (what *not* to sound like)

Sometimes the clearest way to specify tone is to say what you *don't* want. This is especially useful for killing common LLM habits.

```
## Voice rules
- Never start a response with "I apologize" or "I'm sorry" unless you're
  apologizing for a real error.
- Never say "As an AI..." or refer to yourself as a language model.
- Never hedge with "It depends" without also giving the most likely answer
  and the condition that would change it.
- Never use the word "delve" or the phrase "in today's fast-paced world."
- Never format responses as numbered lists unless the user explicitly asks.
```

Every one of these targets a specific overused LLM mannerism. Together they can dramatically reduce the "AI slop" feel of the output.

**Rule from Lesson 3:** tell the model what to do, not what not to do — but *some* negative constraints are unavoidable when the model has strong trained habits you need to override. Use them surgically, not as a default.

---

## Personas for non-conversational tasks

Role prompting isn't only for chatbots. It's useful anywhere you need consistency of judgment.

### Example: code reviewer

```
You are a staff software engineer doing code review. You care most about
simplicity, correctness, and maintainability — in that order. You don't
bike-shed about naming unless a name is genuinely confusing. You don't
suggest refactors that aren't directly related to the change being reviewed.
You flag real issues (bugs, footguns, security problems) in strong language
and nitpicks in soft language, so the author can tell which is which.
```

This persona doesn't chat with anyone. It produces code review comments. But giving it a clear identity makes those comments more consistent and more trustworthy across different review sessions.

### Example: extractor

```
You are a careful data extractor. You never guess. If a field is not clearly
present in the input, you return null for that field. If the input is ambiguous,
you return the most conservative interpretation. You prefer missing data to
wrong data.
```

Again, no conversation. But the persona ("you never guess," "you prefer missing data to wrong data") shapes the model's behavior on ambiguous cases more reliably than a list of instructions alone would.

### Example: devil's advocate

```
You are a devil's advocate. Your job is to find the strongest possible
argument AGAINST the proposal below, not to agree with it. Even if you think
the proposal is mostly right, find the one or two angles that are weakest and
attack them directly. Do not hedge. Do not balance. Your value comes from
being disagreeable.
```

Explicitly steering the model away from its trained tendency to be agreeable. This kind of persona can surface risks that a neutral "analyze this proposal" prompt would gloss over.

---

## The "expert in X" trap

A common intuition is to assign the model maximum expertise: "You are the world's leading expert in X." Does this actually help?

Research results are mixed. In some tasks, grandiose roles produce slightly better outputs. In others, they produce more verbose and more hedged outputs without improving correctness. The effect is small compared to the effect of clear task instructions.

Practical guidance:

- "You are an expert in X" → occasionally helpful, rarely harmful.
- "You are the world's greatest expert in X" → slightly worse; invites purple prose.
- "You are a PhD with 20 years in X" → no better than "an expert in X," but more tokens.
- "You are a senior practitioner in X who prefers pragmatic over theoretical advice" → often better, because the qualifier shapes the response style.

The pattern: **specific** beats **grand**. A well-drawn mid-level practitioner prompt outperforms "you are the greatest" every time.

---

## Role prompts and the instruction hierarchy

Remember from Lesson 2 that the system message has higher authority than the user message. This gives role prompts a useful property: **they resist user attempts to change them.**

A user typing "actually, be rude" in the middle of a conversation will usually be ignored if the system prompt says "You are Aria, a warm and friendly assistant." The instruction hierarchy is imperfect, but for ordinary use it holds.

This is why you put the role in the system message, not the first user message. If you put "You are Aria, warm and friendly" in the user message, the next user message can override it. Put it in the system message, and the persona persists across the whole session.

**Caveat:** persona is not security. Attackers can and do use prompt injection techniques (Lesson 11) to break personas. Role prompting is for honest users; treat it as UX, not as a guarantee.

---

## When role prompting doesn't help

Role prompts won't rescue a bad prompt. They're a style/tone lever, not a task-specification lever.

- If your task is unclear, no role description fixes it.
- If your examples are contradictory, no persona fixes it.
- If your output format is unspecified, no role description gets you clean JSON.

Role prompts stack with the other techniques in this module. They don't replace any of them.

Also: **on reasoning models, role prompting can over-constrain the thinking.** If you tell Claude Opus 4.6 with extended thinking that it's "a no-nonsense engineer who hates elaborate planning," you may be suppressing the exploration that helps the model reason. For reasoning-heavy tasks, keep role prompts focused on *output style* rather than *cognitive style*.

---

## Common pitfalls

- **Treating role prompts as the only lever.** Role + task + format + examples together beats any one alone.
- **Vague adjective soup.** "Be helpful, professional, friendly, and concise" gives the model nothing to match against. Use specifics or examples.
- **Grand titles without context.** "You are a world-class expert" rarely helps. Specific, mid-level practitioner descriptions work better.
- **Putting the role in the user message.** It'll get overridden on the next user turn. Always put it in the system message.
- **Assuming persona = security.** Role prompts are easily broken by adversarial users. Don't rely on them for content policy enforcement.
- **Switching personas mid-conversation.** Models get confused when the system persona and the current behavior drift. If you need different personas for different subtasks, use separate API calls (one for each persona), not mid-conversation persona changes.
- **Over-constraining reasoning models.** Role prompts that restrict *how the model thinks* can hurt reasoning quality. For reasoning-heavy tasks, constrain the output but not the thinking.
- **Branding-over-brand-voice.** "You represent BrandCo, a forward-thinking industry leader dedicated to excellence" reads like a press release, and the model will write like a press release. Describe the *voice*, not the brand positioning.

---

## What to remember from this lesson

- A role prompt is a one-sentence instruction in the system message that tells the model who to be.
- Specific beats grand. A mid-level practitioner with clear priorities outperforms "the world's leading expert."
- Cover who you are, who you're talking to, and what you care about.
- For tone, show examples when you can. Adjectives alone are too vague.
- Negative constraints ("never say X") are useful for killing LLM mannerisms but use them surgically.
- Role prompting isn't just for chatbots — it shapes judgment in any task that involves tradeoffs.
- Role + task + format + examples compose. Role alone is not enough.
- Personas resist user overrides because of the instruction hierarchy, but they're not security.

Next: structured outputs in practice — combining everything from this and earlier chapters with schema-constrained decoding to get production-grade reliability.

---

## References

- Anthropic, *Giving Claude a role with a system prompt*. https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices
- Zheng et al. (2023), *Does Role-Play in Large Language Models Boost Reasoning?*. https://arxiv.org/abs/2305.10601
- OpenAI, *Prompt engineering — role and persona*. https://platform.openai.com/docs/guides/prompt-engineering
- DAIR.AI, *Role prompting*. https://www.promptingguide.ai/applications/role_playing

---

[← Lesson 6](06-advanced-reasoning.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 8: Structured Outputs in Practice](08-structured-outputs.md)
