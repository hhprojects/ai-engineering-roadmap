# Lesson 4 — The Transformer Architecture

> **The single sentence version:** A modern LLM is a stack of identical "blocks," each of which lets every token look at every other token (attention) and then does some independent thinking per token (MLP) — repeated 30-100 times.

You already know the pieces: tokens become embeddings (Lesson 3), positions get added, and then... what? This chapter is the tour of the machine. We'll stay at the architectural level — Lesson 5 goes deep on attention specifically, and Lesson 6 covers training and inference.

---

## Three architecture families, one winner

When you read older papers, you'll see three flavors of transformer:

1. **Encoder-only** (BERT, RoBERTa, DeBERTa). Takes text in, produces a rich representation of it. Great for classification, retrieval, masked language modeling. Not great for generation — it doesn't predict next tokens.
2. **Decoder-only** (GPT, Claude, Llama, Qwen, Mistral, DeepSeek, Gemini — every modern chat model). Takes tokens in, predicts the next token. Autoregressive. This is what you're building with when you call any LLM API today.
3. **Encoder-decoder** (original *Attention Is All You Need* transformer, T5, BART, Whisper, Flan-T5). Two stacks: an encoder that reads the input, a decoder that generates the output. Good for translation, summarization, speech-to-text.

For the rest of this chapter — and most of this course — "transformer" means **decoder-only**. That's the architecture of every frontier chat model and the one you need to understand deeply. Encoder-only and encoder-decoder are legitimate choices for specific tasks, but they're no longer the dominant design.

---

## The decoder-only stack, top to bottom

Here's the full architecture of a modern decoder-only transformer, from input to output:

```
┌─────────────────────────────────────────────────┐
│  Output: logits over vocabulary (shape V)       │   ← what token comes next?
├─────────────────────────────────────────────────┤
│  LM Head (linear projection to vocab size)      │
├─────────────────────────────────────────────────┤
│  Final Layer Norm                               │
├─────────────────────────────────────────────────┤
│  Transformer Block 32  (identical layout)       │   ← repeat 12-100 times
│  Transformer Block 31                           │      depending on model size
│  ...                                            │
│  Transformer Block 2                            │
│  Transformer Block 1                            │
├─────────────────────────────────────────────────┤
│  Input Embedding + Position Info (Lesson 3)     │
└─────────────────────────────────────────────────┘
     ↑
     Token IDs from tokenizer (Lesson 2)
```

Two things matter here:

1. Every transformer block has **the exact same architecture**. They differ only in their learned weights. The depth of the network comes from stacking the same block repeatedly.
2. The output is a vector of **logits** — one number per token in your vocabulary. Softmax that vector, and you have a probability distribution over what comes next. The LM head that produces these logits is often **tied** to the input embedding matrix (GPT-2, Llama) — the same matrix is used to map tokens to vectors on the way in and to map vectors to tokens on the way out. This saves parameters and tends to help training.

---

## Inside one block

Every decoder block does two things, in this order:

```
┌──────────────────────────────────────┐
│   input x                            │
│         │                            │
│         ├──────────┐                 │
│         ▼          │                 │
│   LayerNorm        │                 │
│         │          │                 │
│         ▼          │                 │
│   Self-Attention   │                 │
│         │          │                 │
│         ▼          ▼                 │
│         +──────────┘  residual add   │
│         │                            │
│         ├──────────┐                 │
│         ▼          │                 │
│   LayerNorm        │                 │
│         │          │                 │
│         ▼          │                 │
│   MLP (feed-fwd)   │                 │
│         │          │                 │
│         ▼          ▼                 │
│         +──────────┘  residual add   │
│         │                            │
│         ▼                            │
│     output                           │
└──────────────────────────────────────┘
```

Two sub-blocks:

- **Self-attention** — the "communication" step. Each token looks at every previous token (and itself) and decides what information to pull in. This is the subject of Lesson 5.
- **MLP** (multi-layer perceptron, also called "feed-forward") — the "computation" step. Each token gets processed independently through a small two-layer network. No token looks at any other during this step.

Around each sub-block, two things:

- **Layer Norm** — normalizes the vector to mean 0 and standard deviation 1 before the sub-block runs. Modern models use **RMSNorm** (a simpler variant) and put it *before* the sub-block ("pre-norm"), which makes training more stable.
- **Residual connection** — the sub-block's output is *added* to its input, not replacing it. This creates a "highway" that gradients can flow through during training and is the single most important trick for making deep networks trainable. Without residuals, a 30-layer transformer wouldn't train at all.

### Why alternate attention and MLP?

The canonical intuition, from the *Anthropic Transformer Circuits* research:

- **Attention moves information between positions.** "The cat" and "sat on the mat" are separate parts of the sentence; attention is how "sat" learns that its subject is "cat."
- **MLP transforms information at a position.** Once "sat" has gathered the right context, the MLP is where the model does something useful with it — compute a feature, recognize a pattern, update a belief.

You can think of a transformer as alternating between *"look around"* and *"think about what you saw,"* 30 times in a row. Each block refines the representation of every token slightly. By the final block, each token's vector carries enough information to predict what comes next.

---

## What the MLP is actually doing

The MLP is embarrassingly simple:

```
MLP(x) = W_down · activation(W_up · x)
```

Where:
- `x` has dimension `d_model` (typically 4096 or 8192 for a 7B model)
- `W_up` expands to `d_ff` (typically 4× d_model, so ~16k-32k)
- An activation function (GELU in GPT-2, SwiGLU in Llama, ReGLU in PaLM)
- `W_down` compresses back to `d_model`

That's it. Two linear layers and a nonlinearity. Yet the MLP accounts for roughly **two-thirds of the total parameters** in a modern LLM. Most of what the model "knows" is stored here — facts about the world, linguistic patterns, coding idioms. Attention is the routing mechanism; the MLP is the knowledge store.

Research from the Transformer Circuits community (and similar work at DeepMind) has found specific "neurons" in MLP layers that correspond to specific facts — "Paris is the capital of France" lives in some particular set of rows in some particular MLP somewhere in the middle of the model. You can literally edit these to update facts (the ROME and MEMIT papers).

---

## How big is "big"?

Here's what the architecture looks like for some real models:

| Model | Params | Layers | d_model | Heads | Context | Vocab |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 Small | 124M | 12 | 768 | 12 | 1024 | 50,257 |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 | 1024 | 50,257 |
| Llama 3 8B | 8B | 32 | 4096 | 32 | 128k | 128,256 |
| Llama 3 70B | 70B | 80 | 8192 | 64 | 128k | 128,256 |
| Mixtral 8×7B | 47B | 32 | 4096 | 32 | 32k | 32,000 |
| Karpathy's nanoGPT (Project 3) | ~1M | 4 | 64 | 4 | 256 | ~65 |

Your character-level transformer in Project 3 has the **same architecture** as Llama 3 70B. The difference is scale: ~1 million parameters vs. 70 billion. Same block structure, same attention mechanism, same MLP layout. That's why implementing the small one teaches you so much about the big one.

---

## Why the transformer won

Before transformers, sequence modeling was dominated by LSTMs and GRUs. The transformer paper (2017) introduced two advantages that compound:

1. **Parallel training.** LSTMs have to process tokens one at a time during training (position `t` depends on the hidden state from position `t-1`). Transformers process the entire input simultaneously — every position is computed in parallel during training. This means you can train them far more efficiently on GPUs.
2. **Long-range dependencies.** An LSTM has to carry information through many small state updates to get from position 1 to position 1000; each step risks losing signal. A transformer's attention gives every position direct access to every other position in a single step. No information bottleneck.

These two properties turned out to be the entire game. Every scaling advantage the LLM industry has enjoyed since 2018 traces back to "transformers are the first architecture that made spending 10,000 GPUs on a single model actually work."

---

## Common pitfalls

- **Thinking the transformer "reads left to right."** At training time, all positions are processed simultaneously; the causal mask in attention (Lesson 5) is what enforces that later tokens can't see earlier ones. At inference time, it generates left-to-right, but that's a property of autoregressive *sampling*, not of the architecture itself.
- **Assuming more layers = more smart.** More layers help, but the gains are sub-linear and eventually plateau. Modern scaling is balanced across depth, width (d_model), and training data.
- **Confusing "hidden state" terminology.** The transformer has no recurrent hidden state like an LSTM. What people sometimes call "the hidden state at position t" is just the output vector of some layer at that position. Every position has its own.
- **Overweighting attention.** Attention is the glamorous mechanism, but the MLP holds most of the parameters and most of the knowledge. Don't treat the MLP as a footnote.

---

## What to remember from this lesson

- Modern LLMs are **decoder-only transformers**.
- The architecture is a stack of identical blocks. Each block does attention, then an MLP. Around each sub-block: layer norm and residual connections.
- Attention moves information between positions. MLPs transform information at each position.
- The MLP holds ~2/3 of the parameters. Most knowledge lives there.
- Same architecture scales from a 1M-parameter toy to a 70B production model — only the sizes change.
- Transformers won because they parallelize training and give every position direct access to every other position.

Next chapter: the one mechanism at the heart of all of this. Attention.

---

## References

- Vaswani et al. (2017), *Attention Is All You Need* — the paper that started it all. https://arxiv.org/abs/1706.03762
- Jay Alammar, *The Illustrated Transformer*. https://jalammar.github.io/illustrated-transformer/
- poloclub, *Transformer Explainer* — interactive GPT-2 walkthrough. https://poloclub.github.io/transformer-explainer/
- Anthropic, *Transformer Circuits Thread* — how to think about what each component does. https://transformer-circuits.pub/
- Andrej Karpathy, *nanoGPT*. https://github.com/karpathy/nanoGPT

---

[← Lesson 3](03-embeddings-and-positional-encoding.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 5: Attention](05-attention.md)
