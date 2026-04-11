# Lesson 6 — Training and Inference

> **The single sentence version:** Training a language model means slowly nudging billions of parameters so the model's predicted next tokens match the actual next tokens in a giant pile of text; inference means running the trained model forward to generate new tokens.

As an application engineer, you don't need to train a frontier model from scratch — that's a billion-dollar endeavor. But you *do* need to understand roughly what happens during training (so you can reason about model behavior) and exactly what happens during inference (so you can reason about cost and latency). This chapter covers both.

---

## The training objective

A pre-trained language model has exactly one training objective: given a sequence of tokens, predict the next token. Mathematically, we minimize the **cross-entropy loss**:

```
L = -Σᵢ log P(tokenᵢ | token₀, token₁, ..., tokenᵢ₋₁)
```

That's the sum, over every position in the training data, of the negative log probability the model assigned to the actual next token. If the model was very confident and correct, `log P` is close to 0 (small loss). If it was confidently wrong, `log P` is very negative (big loss). Training is just repeatedly computing this loss and updating the weights to make it smaller.

**The key insight:** this objective is incredibly simple, and you can generate unlimited training data from any text. No labels, no human annotation — just text in, next token prediction out. That's why scale is so powerful here. The internet has trillions of tokens; every single one is a training example.

---

## What "training" actually looks like

Here's the workflow for pre-training a model, stripped to essentials:

1. **Tokenize** your training corpus — often 1-15 trillion tokens, scraped from the web, books, code repositories, and filtered aggressively.
2. **Pack** tokens into fixed-length sequences (e.g., 8192 tokens each), with special tokens marking document boundaries.
3. For each batch of sequences:
   - Forward pass: run the sequence through the model, producing a prediction for every position simultaneously (the causal mask from Lesson 5 is what makes this possible).
   - Compute loss against the actual tokens.
   - Backward pass: compute gradients of the loss with respect to every parameter.
   - Optimizer step: nudge each parameter slightly in the direction that reduces loss. Modern models use **AdamW**, a variant of stochastic gradient descent with momentum and weight decay.
4. **Repeat** for hundreds of thousands or millions of steps.

That's it. No fancy tricks at the high level. The craft is in the details: data quality, learning rate schedules, batch size, mixed-precision numerics, gradient clipping, checkpoint strategy, distributed training across thousands of GPUs. The details are where frontier labs spend their secret sauce.

### Pre-training compute, in scale

To calibrate: training a frontier model in 2024 cost on the order of $50-100 million and required tens of thousands of H100 GPUs running for months. The 2017 original transformer was trained on 8 GPUs for a few days. The compute involved has grown roughly 10,000× in seven years. This is the scaling that underlies every capability jump.

---

## Post-training: turning a predictor into an assistant

A pre-trained model predicts next tokens well but doesn't behave like an assistant. To fix that, labs run **post-training**, usually in three stages:

### 1. Supervised Fine-Tuning (SFT)

Collect tens to hundreds of thousands of example conversations where a human writes a good assistant response. Train the model on those examples — same cross-entropy loss, same optimizer, but now the training data is "question → ideal response" instead of raw internet text.

This gives the model a sense of what an assistant *should* sound like. SFT is what makes "Tell me a joke" produce a joke instead of continuing a Reddit thread about jokes.

### 2. Reinforcement Learning from Human Feedback (RLHF)

SFT teaches the model to imitate; RLHF teaches it to prefer better responses over worse ones. Roughly:

1. Give the model a prompt, have it generate two (or more) completions.
2. Have a human rank them.
3. Train a **reward model** (another, smaller model) to predict those rankings.
4. Use the reward model as a signal to further train the main model — rewarding it when it produces responses the reward model scores highly.

The optimization is usually done with **PPO** (Proximal Policy Optimization) or, more recently, **DPO** (Direct Preference Optimization), which skips the reward model entirely and optimizes directly on pairwise preferences.

### 3. Constitutional AI / RLAIF

Having humans rank millions of responses is expensive. Anthropic pioneered using a strong LLM as the judge instead of humans — the model critiques its own outputs against a written "constitution" of rules, and those critiques drive the optimization. This is how Claude is trained to refuse harmful requests without relying on massive human labeling.

The end result of SFT + RLHF (or its variants) is the model you talk to when you use ChatGPT or Claude. The base predictor is still in there — post-training just teaches it which predictions to prefer.

---

## Inference: how generation actually works

Once training is done, the model is frozen. Every API call you make goes through the same two-phase process.

### Phase 1: Prefill

The model reads your input prompt in parallel. Every token's attention, MLP, and residuals are computed simultaneously. For a 1000-token prompt, this is one big forward pass through the entire network.

Prefill is **compute-bound** — you're doing matrix multiplications proportional to the input length. It's fast per token (you can prefill thousands of tokens in the time it takes to generate one), but it's what dominates cost for long input, short output tasks like classification.

### Phase 2: Decode

Now the fun part. The model generates tokens one at a time:

1. The model's output at the last position is a logits vector of size `vocab_size`.
2. Apply softmax → probability distribution over next tokens.
3. Sample one token (greedy, temperature, top-p — see Lesson 7).
4. Append that token to the sequence.
5. Run the model again, but *only for the new token's position* — everything before it hasn't changed.

Step 5 is where the **KV cache** becomes critical. Remember that attention needs K and V vectors for every token the new token might want to look at. Those are the same K and V vectors we already computed during prefill for every earlier token. Why recompute them? We don't — we cache them.

```
KV cache shape: [num_layers, 2 (K and V), num_heads, seq_len, head_dim]
```

For a 70B model at 128k context, this cache can be tens of GB. It's the reason long-context inference is memory-hungry — not the weights themselves, but the growing KV cache for whatever conversation you're in.

**Decode is memory-bandwidth-bound.** You're reading 70 billion weights from HBM for *each* token you generate, so the bottleneck is how fast you can stream weights into the compute units. This is why inference on a consumer GPU is slow even if the model fits — consumer GPUs have much lower memory bandwidth than datacenter GPUs.

### Why output tokens cost 4-5× more than input tokens

Prefill processes many tokens in parallel with one pass through the network. Decode processes tokens *one at a time*, each requiring a full forward pass through 70B weights. The compute per output token is much higher than per input token. Plus, providers have less ability to batch decode requests efficiently (different users are generating different tokens at different rates).

That's why the pricing almost universally looks like:

```
Input:  $3 per million tokens
Output: $15 per million tokens
```

Internalize that ratio. It shapes every cost decision you'll make as an application engineer.

---

## Fine-tuning: the middle ground

Between "use a pre-trained model as-is" and "train one from scratch" is **fine-tuning**. You take a base model (often one you didn't train) and run additional training on your own data. Typical uses:

- **Domain adaptation** — fine-tune a general model on medical or legal text to improve performance on those domains.
- **Task specialization** — fine-tune a chat model to always output JSON in a particular schema.
- **Style imitation** — fine-tune on a writer's previous work to mimic their voice.
- **Instruction following on new formats** — teach the model a new kind of query it's never seen.

Full fine-tuning updates all the model's weights and is expensive. **LoRA** (Low-Rank Adaptation) is the standard efficient alternative: freeze the original weights and train a small set of "adapter" matrices that get added to the originals. Typical LoRA adapters have 0.1-1% as many parameters as the full model, making fine-tuning possible on a single GPU and making it practical to have many specialized adapters for one base model.

Most engineering teams never fine-tune anymore — prompt engineering, RAG (Module 3), and tool use cover the majority of use cases at a fraction of the operational complexity. Reach for fine-tuning when prompting has provably failed for your task.

---

## What to remember from this lesson

- Training a language model is minimizing cross-entropy loss on next-token prediction, at massive scale.
- Post-training (SFT + RLHF) is what turns a predictor into an assistant.
- Inference has two phases: prefill (parallel, compute-bound) and decode (sequential, bandwidth-bound).
- The KV cache is what makes sequential decode efficient — we don't recompute K and V for tokens we already processed.
- Output tokens cost 4-5× more than input tokens because decode is sequential and memory-bound.
- Fine-tuning is a powerful tool but rarely your first choice; start with prompting and RAG.

Next chapter: the sampling decisions that happen at the very end of the generation loop.

---

## References

- OpenAI, *Scaling Laws for Neural Language Models* (Kaplan et al., 2020). https://arxiv.org/abs/2001.08361
- DeepMind, *Training Compute-Optimal Large Language Models* (Chinchilla, Hoffmann et al., 2022). https://arxiv.org/abs/2203.15556
- Ouyang et al. (2022), *Training language models to follow instructions with human feedback* (InstructGPT/RLHF). https://arxiv.org/abs/2203.02155
- Bai et al. (2022), *Constitutional AI: Harmlessness from AI Feedback*. https://arxiv.org/abs/2212.08073
- Rafailov et al. (2023), *Direct Preference Optimization*. https://arxiv.org/abs/2305.18290
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*. https://arxiv.org/abs/2106.09685
- Pope et al., *Efficiently Scaling Transformer Inference* (the KV cache analysis). https://arxiv.org/abs/2211.05102

---

[← Lesson 5](05-attention.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 7: Sampling](07-sampling-strategies.md)
