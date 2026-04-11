# Lesson 3 — Embeddings and Positional Encoding

> **The single sentence version:** The model can't read your tokens — it can only do math on vectors — so the first layer looks each token up in a big table and adds a "position stamp" so the network knows which token came first.

Tokens are integers. Neural networks are matrix multiplications. We need a bridge between them. That bridge is **embeddings**: a lookup table that turns every token ID into a dense vector. And because the transformer processes all tokens in parallel, we need to separately tell it *where* each token sits in the sequence — that's **positional encoding**.

This chapter explains both, and why they're more interesting than they sound.

---

## Why a lookup table?

Suppose your tokenizer's vocabulary has 50,257 entries (that's GPT-2's size). You could one-hot encode each token — a vector of 50,257 zeros with a single 1 — but that's wasteful: 50,257 dimensions to represent one token, with zero structure between tokens.

Instead, we learn a matrix `E` of shape `[vocab_size, d_model]`, where `d_model` is the model's hidden dimension (typically 768, 2048, 4096, 8192 depending on model size). Token ID `42` becomes `E[42]` — one row from the matrix. That row is a **dense vector**: every dimension carries information, and two tokens with similar meanings end up with similar vectors *because the training objective nudges them that way*.

GPT-2's embedding matrix has 50,257 × 768 ≈ **39 million parameters**. A modest 7B model like Llama 3 8B has a 128,256 × 4096 ≈ **525 million parameter** embedding table — a sizable chunk of the whole model.

---

## What the embedding vector "means"

The usual claim is that embeddings capture semantic meaning — words with similar meanings have similar vectors. This is mostly true, but with nuance:

- Early in training, embeddings are random.
- As the model learns to predict next tokens, it's forced to position words with similar distributional behavior near each other in embedding space. "King" and "queen" end up near each other because they appear in similar contexts. So do "Paris" and "London".
- The famous *word2vec* analogies — `king - man + woman ≈ queen` — work (partially) because of this geometric structure. Transformer embeddings show similar patterns, but less cleanly than word2vec because they're trained for a different objective.
- Embeddings are **contextualized** by later layers. The embedding for "bank" starts ambiguous (river bank vs. financial bank); by the time the token has passed through 30 attention layers, its representation has been pushed toward one meaning or the other based on surrounding tokens.

So the embedding table is the *starting point*, not the final representation. It's where context-free meaning lives before attention adds context.

---

## The position problem

Transformers don't process tokens left-to-right like an RNN. They process all tokens **in parallel** — that's the whole reason they train faster than the LSTMs that preceded them. But parallelism comes with a cost: without extra help, the transformer has no way to know that "dog bites man" is different from "man bites dog." Swap any two words and the model sees the same unordered soup of vectors.

We fix this by injecting position information directly into the input. The original 2017 transformer paper (*Attention Is All You Need*) used a fixed sinusoidal encoding. Modern models have mostly moved to learned or rotational approaches. You should know all three.

### Sinusoidal positional encoding (original transformer)

For each position `pos` and each dimension `i` of the embedding vector, compute:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

You add this matrix element-wise to the token embeddings *before* the first transformer layer. The low dimensions oscillate slowly (giving you a coarse position signal), the high dimensions oscillate quickly (giving you a fine position signal). A model trained on length-512 sequences can, in principle, generalize to longer sequences because the sinusoids extrapolate smoothly.

**Why sine and cosine?** Because the dot product of positions at a fixed offset is constant, which means the model can learn to attend to "the token two positions back" regardless of where in the sequence it is. It's a mathematically elegant solution.

### Learned positional embeddings

GPT-2 and BERT took a simpler route: treat position like a token, give each position its own learned embedding, and add it to the token embedding.

```
input_to_layer_1[pos] = token_embedding[token_id] + position_embedding[pos]
```

This works well in practice, but it has one big drawback: **the model cannot handle positions it never saw during training**. A GPT-2 trained on 1024-token sequences has no position embedding for position 1025, and asking it to process a longer sequence is undefined behavior.

### Rotary Position Embedding (RoPE) — the current default

Introduced by the RoFormer paper (2021) and now used by **Llama, Qwen, Mistral, Gemma, DeepSeek**, and most open-weight models released after 2022. Instead of adding position information to the embeddings, RoPE *rotates* the query and key vectors inside the attention mechanism by an angle that depends on position. The rotation is applied in 2D subspaces of the vector.

Why this is clever:

- Two tokens at positions `m` and `n` end up with a dot product that depends only on `m - n`, the *relative* distance. This matches how language actually works — what matters for attention is "how far apart are these words," not their absolute position.
- RoPE extrapolates to longer sequences much better than learned embeddings, though still imperfectly. Techniques like **YaRN** and **RoPE scaling** stretch a RoPE-trained model to 4× its original context with only minor degradation.
- It composes cleanly with KV caching and flash attention, which is why every serious open-weight model adopted it.

You don't need to implement RoPE to understand LLMs, but you do need to recognize the name — it's the "how do we handle long context" answer of the 2023+ era.

### ALiBi — the underdog

**Attention with Linear Biases** skips position encodings entirely and instead adds a small negative bias to attention scores between tokens that are far apart. Used by BLOOM and MPT. Less common today but worth knowing as the "stripped-down" alternative to RoPE.

---

## Putting it together: the input to layer 1

Here's what actually flows into the first transformer block, step by step, for a 6-token input:

```
1. Tokenize:       "The cat sat on the mat"  →  [464, 3797, 3332, 319, 262, 2603]
2. Look up:        shape [6, d_model] token embeddings
3. Add position:   shape [6, d_model] with position info baked in
4. Feed to layer 1
```

That "add position" step is either literal element-wise addition (sinusoidal, learned) or a rotation applied inside attention (RoPE). Either way, the network now has enough information to know "dog bites man" ≠ "man bites dog."

---

## Common pitfalls

- **Thinking embeddings encode words, not tokens.** They encode tokens. If your tokenizer splits "unfortunately" into `["un", "fortunate", "ly"]`, you're looking up three separate embeddings and letting the model compose meaning from them.
- **Assuming positional embeddings are trivial.** They're a major reason why early GPT models couldn't handle long documents — learned embeddings simply don't exist for positions beyond the training length. Context window extension is mostly about fixing the position encoding.
- **Confusing token embeddings with sentence embeddings.** A sentence embedding (for semantic search, clustering, etc.) usually comes from pooling over token embeddings *after* they've passed through the whole model. The raw embedding table isn't what you want for retrieval — the *output* of the final layer, possibly mean-pooled, is.

---

## What to remember from this lesson

- Embeddings are a learned lookup table that maps token IDs to dense vectors.
- The embedding matrix is often the single largest parameter group in a small model.
- Token embeddings start context-free; later layers refine them using surrounding tokens.
- Transformers process tokens in parallel, so they need explicit position information.
- The three approaches you'll see: sinusoidal (original), learned (GPT-2, BERT), and RoPE (modern Llama family). Modern long-context models almost all use RoPE with some scaling technique.
- A model's effective context length is tied to its position encoding — extending it is a real engineering problem, not a toggle.

---

## References

- Vaswani et al., *Attention Is All You Need* (original transformer paper with sinusoidal PE). https://arxiv.org/abs/1706.03762
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*. https://arxiv.org/abs/2104.09864
- Press, Smith, Lewis, *Train Short, Test Long: ALiBi*. https://arxiv.org/abs/2108.12409
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*. https://arxiv.org/abs/2309.00071
- Jay Alammar, *The Illustrated Transformer*. https://jalammar.github.io/illustrated-transformer/

---

[← Lesson 2](02-tokenization.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 4: Transformer Architecture](04-transformer-architecture.md)
