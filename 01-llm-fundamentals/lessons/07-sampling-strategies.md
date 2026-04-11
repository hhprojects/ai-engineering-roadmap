# Lesson 7 — Sampling Strategies

> **The single sentence version:** The model gives you a probability distribution over every possible next token; sampling is how you turn that distribution into an actual choice, and the choice you make dramatically affects output quality, creativity, and reliability.

You've seen the generation loop: at each step, the model produces a logits vector of size `vocab_size`, softmax turns it into a probability distribution, and... you pick one. That picking step is sampling. It's the most user-visible parameter surface in the whole LLM API, and it's the parameter most people tune without really understanding.

This chapter explains every sampling strategy you'll see in an API, when each one is the right choice, and what the knobs actually do.

---

## The raw material: logits and probabilities

At the end of every forward pass, the model outputs a vector of logits — one real number per token in the vocabulary:

```
logits = [2.3, -1.4, 5.1, 0.8, -3.2, ..., 4.7]    # shape [vocab_size]
```

These are just "scores" — bigger means "more likely to come next." They are unnormalized and can be any real number. To turn them into probabilities, we apply softmax:

```
probabilities = exp(logits) / sum(exp(logits))      # sums to 1
```

Now you have a probability distribution. The question is: how do you pick one token from it?

---

## Greedy decoding: pick the most likely token

The simplest strategy. At each step, pick the token with the highest probability:

```python
next_token = argmax(probabilities)
```

Deterministic. Fast. Totally reproducible. Also: **surprisingly bad for long-form generation**. Greedy decoding tends to produce repetitive, bland text — the model gets stuck in local loops like *"the the the"* or keeps repeating the same phrase every paragraph.

**When to use greedy:** short answers where there's objectively one right answer. "What is 2+2?", "Extract the phone number from this text", factual Q&A. Anything where you'd want the same answer every time.

**When not to use greedy:** creative writing, brainstorming, long-form anything.

---

## Temperature: scaling the distribution

Temperature is a single knob that makes the distribution more or less peaky before sampling from it. You divide the logits by `T` before the softmax:

```python
probabilities = softmax(logits / T)
```

- **T = 1.0** — no change. Sample from the distribution as-is.
- **T < 1.0** (e.g., 0.3) — peaks get sharper. The most likely token becomes *even more* likely; unlikely tokens become almost impossible. At `T → 0`, this converges to greedy.
- **T > 1.0** (e.g., 1.5) — peaks flatten. Unlikely tokens get more probability, creating more diverse (and sometimes more unhinged) outputs. At `T → ∞`, you're sampling uniformly at random from the vocabulary.

Intuition: think of temperature as "how confident should the model be in its top picks?" Low temperature is conservative — stay close to the best bet. High temperature is exploratory — take more risks.

**Typical values:**
- `0.0` — deterministic, effectively greedy. Use for tests, structured extraction, math.
- `0.3` — conservative but not deterministic. Code generation, technical answers.
- `0.7` — the default in most APIs. General-purpose chat.
- `1.0-1.2` — creative writing, brainstorming, poetry.
- `> 1.5` — rarely useful; usually produces incoherent output.

---

## Top-k sampling: truncate the long tail

Temperature alone has a problem. Even with `T = 0.7`, there are ~50,000 tokens in the vocabulary, and the long tail of very unlikely tokens still has some nonzero probability. If you're unlucky, you occasionally sample a completely nonsensical token, and once you do, the rest of the generation degrades (the model is now conditioning on garbage).

**Top-k** fixes this by keeping only the `k` most likely tokens and zeroing out everything else before sampling. `k = 50` is common.

```python
keep the top k tokens by probability
set everything else to 0
renormalize
sample from the pruned distribution
```

Pros: cuts out the nonsense tail.
Cons: `k` is a hard constant that doesn't adapt. If the distribution is very peaky (model is super confident about the top 5), keeping 50 still admits junk. If the distribution is flat (model is uncertain among 500 candidates), keeping 50 throws away legitimate options.

Top-k has mostly been replaced by top-p for this reason.

---

## Top-p (nucleus) sampling: adaptive truncation

**Top-p**, also called nucleus sampling, truncates based on *cumulative probability* instead of a fixed count.

```
Sort tokens by probability, descending.
Keep adding them to the "nucleus" until their cumulative probability reaches p (e.g., 0.9).
Sample from the nucleus.
```

If the model is confident (`P(best) = 0.85`), the nucleus might be just 2-3 tokens. If the model is uncertain, the nucleus might be 50-100 tokens. The set adapts to the model's confidence.

Top-p is the default sampling strategy in almost every modern API. Default values are usually `p = 0.9` or `p = 0.95`.

**Common combinations:**
- Temperature 0.7 + top-p 0.9 → the OpenAI and Anthropic defaults. Balanced creativity.
- Temperature 0 (greedy) → top-p is irrelevant. You're picking the single best token.
- Temperature 0.2 + top-p 0.1 → very conservative; essentially greedy with a tiny safety valve.

---

## Other knobs you'll see

### `frequency_penalty` and `presence_penalty` (OpenAI)

Both are post-hoc adjustments to the logits based on how often a token has already appeared in the output:

- **Frequency penalty** subtracts an amount *proportional to how many times* a token has appeared. Used to reduce repetition.
- **Presence penalty** subtracts a fixed amount *if the token has appeared at all*. Used to push the model toward new topics.

Typical values: 0.0 to 1.0. Higher values mean more aggressive penalization. Mostly useful to fix specific repetition problems; don't crank them by default.

### `repetition_penalty` (most open-source model servers)

Similar idea, but multiplies the logit by a factor instead of subtracting. `repetition_penalty = 1.1` is a common default for open-source models.

### `min_p` (newer alternative to top-p)

Keeps tokens whose probability is at least `min_p × P(top_token)`. Scales the cutoff relative to the top token, so it's more robust to very peaked or very flat distributions. Supported in llama.cpp, vLLM, and some newer APIs.

### `seed`

Locks the random number generator so you get the same sampling outcome on repeated calls. Useful for reproducibility in tests, not for production (you still want diversity there).

---

## Beam search (mostly obsolete for chat models)

Instead of sampling one token at a time, beam search maintains the top `k` candidate *sequences* and extends all of them in parallel, pruning back to the top `k` after each step. Popular in translation and older NLG systems because it finds globally better sequences than greedy.

Why it's not used for chat models: beam search tends to produce bland, generic text that's locally probable but globally repetitive. It also scales badly with long generations. The frontier has moved decisively to sampling.

You'll still see beam search in:
- Whisper (speech-to-text)
- Some translation models
- Domain-specific structured generation where you can prune with a grammar

But not in GPT, Claude, or Llama inference.

---

## Logprobs: what the model was actually thinking

Most APIs let you request **logprobs** — the log probability of each generated token, plus optionally the top-N alternatives and their logprobs. This tells you how confident the model was.

Why this matters:

- **Confidence estimation.** If the model's chosen token had probability 0.95, it's confident. If 0.15, it's guessing. You can use this to flag uncertain answers for human review.
- **Debugging generation quality.** If you see the model getting consistently low logprobs on a prompt, you're probably prompting it poorly.
- **Classification via generation.** If you ask the model "Is this review positive or negative?" and get back the logprobs of "positive" vs "negative", you can use them as a soft classifier score instead of parsing text.
- **Evaluation.** Perplexity (a standard LM quality metric) is computed from logprobs.

Request them with:
- OpenAI: `logprobs=True, top_logprobs=5`
- Anthropic: not directly exposed in the public API as of early 2026
- Local models (vLLM, llama.cpp, transformers): always available

---

## Putting it together: the sampling decision tree

You're about to make an API call. How should you set these parameters?

1. **Is there one correct answer?** (Math, classification, structured extraction)
   → `temperature = 0` (greedy). Ignore top-p. Be deterministic.

2. **Is correctness important but some variation is fine?** (Code generation, technical writing)
   → `temperature = 0.2-0.4, top_p = 0.9`. Low temperature, standard top-p.

3. **Is this general-purpose chat / Q&A?** (Customer support, research assistant)
   → Leave defaults. Probably `temperature = 0.7, top_p = 0.9`. Don't overthink it.

4. **Is this creative or brainstorming?** (Story writing, ideation)
   → `temperature = 0.9-1.2, top_p = 0.95`. More diversity.

5. **Are you debugging or writing evals?**
   → `temperature = 0` and request logprobs. You want determinism and visibility.

6. **Are you running a benchmark?**
   → Whatever the benchmark specifies. Usually `temperature = 0`.

---

## Common pitfalls

- **Treating `temperature` as a "smartness" knob.** It's not. Higher temperature doesn't make the model think harder; it makes it pick less probable tokens. Often less probable means less correct.
- **Assuming `temperature = 0` is fully deterministic.** It's *mostly* deterministic, but floating-point nondeterminism in batched inference can still produce different outputs across runs. For true determinism, you also need a fixed seed and the same hardware configuration.
- **Trying to fix repetition with temperature.** Usually repetition is a symptom of a bad prompt or a bad model, not a sampling issue. Crank `frequency_penalty` only after you've ruled out the prompt.
- **Forgetting that different APIs do sampling slightly differently.** The "temperature 0.7" behavior on OpenAI is not identical to "temperature 0.7" on Anthropic. Always A/B test when you port prompts.
- **Conflating logprobs with confidence.** The model can be high-logprob wrong and low-logprob right. Logprobs are a signal, not a truth. Use them as an input to uncertainty, not as uncertainty itself.

---

## What to remember from this lesson

- The model produces logits; softmax produces a probability distribution; sampling picks one token.
- Greedy (`T = 0`) is for deterministic tasks. Everything else is some form of randomized sampling.
- Temperature scales the distribution. Top-p truncates the long tail adaptively.
- Default: `temperature = 0.7, top_p = 0.9` works for most chat use cases. Go lower for correctness-critical tasks, higher for creative ones.
- Logprobs are free visibility into the model's confidence and should be used more than they are.

This closes out the "how does it work" half of the course. The next chapters — model families, reasoning models, structured outputs, pricing — are about how to use what you just learned in real systems.

---

## References

- Holtzman et al. (2019), *The Curious Case of Neural Text Degeneration* (the top-p paper). https://arxiv.org/abs/1904.09751
- Fan et al. (2018), *Hierarchical Neural Story Generation* (top-k). https://arxiv.org/abs/1805.04833
- Welleck et al. (2019), *Neural Text Generation with Unlikelihood Training*. https://arxiv.org/abs/1908.04319
- poloclub, *Transformer Explainer* — interactive view of how temperature affects the output distribution. https://poloclub.github.io/transformer-explainer/
- Simon Willison, notes on logprobs in OpenAI's API. https://simonwillison.net/tags/logprobs/

---

[← Lesson 6](06-training-and-inference.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 8: Model Families](08-model-families.md)
