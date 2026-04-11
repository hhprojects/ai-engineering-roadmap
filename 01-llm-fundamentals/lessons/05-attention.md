# Lesson 5 — Attention

> **The single sentence version:** For every token, the model computes three vectors — a query, a key, and a value — and uses the query to search over all other tokens' keys to decide which values to pull in.

Attention is the mechanism that made transformers possible and that every chat model uses. Most explanations are either too hand-wavy ("each token decides what to look at") or too equation-heavy (a page of linear algebra). This chapter tries to land in the middle: enough math that you could implement it, enough intuition that you'd remember it a year from now.

---

## The problem attention solves

Consider the sentence:

> "The trophy didn't fit in the brown suitcase because **it** was too big."

What does "it" refer to? Humans know instantly — the trophy, because suitcases being "too big" wouldn't prevent fitting. Now consider:

> "The trophy didn't fit in the brown suitcase because **it** was too small."

Here, "it" is the suitcase. The disambiguating word is `big` vs. `small`, which is seven words away from `it`. Whatever mechanism the model uses to compute the representation of `it` needs to be able to reach back and pull in information from distant, relevant words.

This is the core problem of sequence modeling: how does information at position N incorporate information from positions 1 through N-1? RNNs and LSTMs solved it by passing a running hidden state forward one token at a time, which leaks signal over long distances. Attention solves it by giving every position **direct, one-hop access** to every other position in the sequence.

---

## Query, Key, Value: the search engine analogy

Forget linear algebra for a moment. Think of attention as a database query.

- Every token produces a **query** vector — "what information do I need right now?"
- Every token also produces a **key** vector — "what information am I offering?"
- Every token also produces a **value** vector — "here's the actual information, if you want it."

When we're computing the output for token `it`, we:

1. Take `it`'s query vector.
2. Compute its dot product with *every* token's key vector. That gives us a score per token — how well does my query match your key?
3. Softmax those scores so they sum to 1. Now we have weights — "pay 80% attention to `trophy`, 15% to `big`, 2% to `suitcase`, and 3% scattered everywhere else."
4. Multiply each token's value vector by its weight, sum them all up. That weighted sum is the output of attention for `it`.

That's the whole mechanism. Every token does this in parallel. Every token is simultaneously a searcher (with its query) and a searchable entry (with its key and value).

---

## The math, one equation

Here's the famous formula from the transformer paper:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

Unpacking:

- **Q, K, V** are matrices of shape `[sequence_length, d_k]`. One row per token.
- `Q · Kᵀ` produces a `[seq_len, seq_len]` score matrix: cell `(i, j)` is how much token `i`'s query matches token `j`'s key.
- **Divide by `√d_k`** to keep scores from blowing up as `d_k` grows. Without this, softmax would produce extremely peaky distributions (one token gets weight ≈1, all others ≈0), which makes gradients tiny. In the original paper, `d_k = 64`, so you divide by 8. This is called **scaled dot-product attention**.
- **Softmax along each row** turns scores into probabilities — weights that sum to 1.
- **Multiply by V** — each row of the output is a weighted sum of value vectors.

Output is a `[seq_len, d_k]` matrix: one "contextualized" vector per input token.

---

## Where Q, K, V come from

Each token's input vector `x` (from the previous layer) is multiplied by three learned matrices:

```
Q = x · W_Q   where W_Q has shape [d_model, d_k]
K = x · W_K   where W_K has shape [d_model, d_k]
V = x · W_V   where W_V has shape [d_model, d_v]
```

So a single attention layer has three learnable matrices per head. The model *learns* what it means to "query" and "offer information" — during training, the Q/K/V projections get shaped so that queries and keys line up when they should, and values carry the right information when they're retrieved.

**An important subtlety:** Q, K, and V are all derived from the same input `x`. That's why it's called **self-attention** — every token queries, offers, and supplies all at once, from the same source. When you see "self-attention" vs. "cross-attention," the difference is: self-attention uses the same input for all three; cross-attention (used in encoder-decoder models) takes Q from one source and K/V from another.

---

## Multi-head attention

One attention calculation is fine, but it forces all tokens to share a single "reason to look at each other." Multi-head attention runs the whole mechanism in parallel, multiple times, with different weight matrices each time.

Concretely, if `d_model = 768` and we want **12 heads**, we:

1. Split the 768-dim input into 12 chunks of 64 dims each.
2. Run a separate Q/K/V projection (with separate weights) for each chunk → 12 independent attention calculations.
3. Each head produces a `[seq_len, 64]` output.
4. Concatenate the 12 outputs back into a `[seq_len, 768]` matrix.
5. Multiply by a final output matrix `W_O` to mix the heads back together.

Why? Different heads can learn to attend for different reasons. Interpretability research has shown heads that specialize in things like:

- **Coreference** — this "it" points to that noun.
- **Previous token** — always attends to the immediately preceding token.
- **Syntactic dependencies** — subjects attend to verbs, adjectives attend to their nouns.
- **Position-based** — attend to the token exactly 5 positions back.
- **Content-based** — attend to any earlier token matching a specific pattern.

You get a richer, more factorized representation than any single attention calculation could produce. Modern models use anywhere from 8 (GPT-2 base) to 128 (Llama 3 405B) heads.

---

## Causal masking: no peeking

When a decoder-only model generates text, token at position `t` must only see tokens at positions `0` through `t-1`. It can't see the future — otherwise training would be trivial (the model could just copy).

We enforce this with a **causal mask**: before applying softmax, set the scores for "future" positions to `-∞`. After softmax, those positions get weight 0, so they contribute nothing to the output.

The mask is a triangular matrix:

```
Position:   0    1    2    3
Pos 0:      0   -∞   -∞   -∞
Pos 1:      0    0   -∞   -∞
Pos 2:      0    0    0   -∞
Pos 3:      0    0    0    0
```

This matters more than you'd think — it's what makes the same architecture able to predict *every* next-token simultaneously during training. Input a 1000-token sequence, compute all 1000 "what comes next" predictions in parallel, compare each to the actual next token, backprop on all 1000 losses at once. Without the mask, positions would cheat by looking at their own answers.

---

## The variants you'll actually encounter

Plain multi-head attention is what the 2017 paper proposed. Modern models almost always use one of these tweaks:

### Multi-Query Attention (MQA)

Used by: PaLM, Falcon.

Instead of having separate K and V projections per head, *all heads share the same K and V*. Only Q is per-head. This dramatically reduces the memory needed for the KV cache during inference (Lesson 6), making long-context generation cheaper. The trade-off is slightly worse quality.

### Grouped-Query Attention (GQA)

Used by: Llama 2, Llama 3, Mistral, Qwen, most modern open models.

A middle ground. You have, say, 32 query heads but only 8 K/V heads — each K/V head is shared across a group of 4 query heads. Almost all the speed benefit of MQA, almost all the quality of full multi-head attention. GQA is the standard in 2026.

### Flash Attention

Not a different algorithm — the same math, but an implementation trick that avoids materializing the big `[seq_len, seq_len]` attention matrix in GPU memory. It fuses the whole operation into a single kernel that streams through memory efficiently. Gives 2-4× speedup and lets you train on much longer sequences without running out of GPU memory. Every serious training run since 2023 uses FlashAttention or its descendants.

### Sliding Window Attention

Used by: Mistral, some Gemini variants.

Instead of attending to all previous tokens, only attend to the last N (e.g., the last 4096). Trades perfect long-range access for linear memory cost. Combined with a few "global" attention heads that *do* see everything, this gets you most of the way to long context with a fraction of the compute.

---

## What attention *is not* doing

A few common misconceptions:

- **Attention is not "attention" in the human cognitive sense.** The model is not consciously deciding what's important. It's multiplying matrices; the behavior we call "attention" is an emergent property of trained weights.
- **Attention is not retrieval.** It doesn't go "fetch" information from a database. It produces a weighted sum — a soft blend of all token values. The peakier the softmax, the more it resembles a lookup; the flatter, the more it resembles averaging.
- **Attention is not the only way to move information.** State Space Models (Mamba, Mamba-2) and linear attention variants achieve similar effects with different mechanics. Hybrid models (e.g., Jamba) mix attention blocks and state-space blocks. Attention is dominant but not mandatory.
- **Attention is expensive at long contexts.** The `Q · Kᵀ` matrix is `O(n²)` in sequence length. A 32k-context model spends a huge fraction of its compute on attention. This is the fundamental scaling challenge that Flash Attention, sliding windows, and Mamba-style models are all trying to address.

---

## What to remember from this lesson

- Every token produces a query, key, and value vector.
- Queries are compared against all keys; the softmax of those scores weights the values.
- Multi-head attention runs this in parallel with different projections so different heads can specialize.
- Causal masking prevents decoder models from seeing the future — it's how training on full sequences still teaches next-token prediction.
- Modern models use GQA (not plain MHA) and Flash Attention for efficiency.
- Attention is `O(n²)` in sequence length — the main reason long context is expensive.

You now understand the single most important mechanism in every modern LLM. The rest of the course builds on top of this.

---

## References

- Vaswani et al. (2017), *Attention Is All You Need* (Section 3.2 for attention details). https://arxiv.org/abs/1706.03762
- Jay Alammar, *The Illustrated Transformer* (the scaled dot-product walkthrough). https://jalammar.github.io/illustrated-transformer/
- Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need* (MQA). https://arxiv.org/abs/1911.02150
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models*. https://arxiv.org/abs/2305.13245
- Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. https://arxiv.org/abs/2205.14135
- Anthropic, *A Mathematical Framework for Transformer Circuits*. https://transformer-circuits.pub/2021/framework/index.html

---

[← Lesson 4](04-transformer-architecture.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 6: Training and Inference](06-training-and-inference.md)
