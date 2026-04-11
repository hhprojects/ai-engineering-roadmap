# Project 3 — Mini Transformer from Scratch

🟠 **Advanced** · ~8-12 hours · free (runs on your CPU or laptop GPU)

Nothing beats building it yourself. Implement a transformer from scratch in PyTorch, train it on Shakespeare, and watch it learn to generate text one character at a time. When you finish this, you will never be confused about attention, embeddings, or layer norm again — because you'll have typed them out with your own hands.

This project follows the approach Andrej Karpathy made famous in his "Let's build GPT from scratch" video. The point isn't to outperform GPT-4 (you won't). The point is to fully demystify what's happening inside the thing.

---

## Prerequisites

- Finished **Lessons 1 through 7** — particularly 3 (embeddings), 4 (architecture), 5 (attention), 6 (training/inference), and 7 (sampling)
- Python 3.11+
- PyTorch installed (`pip install torch`) — CPU-only is fine
- Comfortable with numpy and basic linear algebra (matrix multiplication, dot products)
- **Watch Karpathy's video first.** https://www.youtube.com/watch?v=kCc8FmEb1nY — 2 hours. Seriously, watch it *before* you open your editor.

---

## What you'll build

A character-level language model built from scratch in PyTorch, trained on ~1MB of Shakespeare text. No pre-built attention module — you'll implement it yourself. By the end, you'll have:

- A model that generates Shakespeare-ish text (it won't be coherent, but it'll *look* like Shakespeare at a glance)
- A training loop you wrote yourself with visible loss curves
- Enough intuition to read the PyTorch source of any production transformer and understand what's happening
- A small codebase you can extend — add RoPE, add GQA, experiment with activations, compare sampling strategies

---

## What you'll learn

- Implementing **self-attention** from scratch with Q, K, V projections
- Implementing **multi-head attention** by composing single-head attention
- Implementing a **transformer block** with residual connections and layer norm
- Writing a **training loop** with AdamW, gradient clipping, and loss tracking
- **Character-level tokenization** (the simplest possible tokenizer)
- How training loss actually decreases over steps (and what bumps in the curve mean)
- Sampling with **temperature** and **top-k** from a real distribution
- How model size (layers, heads, embedding dim) affects sample quality

---

## Tech stack

- **Python 3.11+**
- **PyTorch** — CPU or GPU. No CUDA required; a laptop CPU trains the small config in ~20 minutes.
- **numpy** — data loading
- **matplotlib** — loss curves
- **tqdm** — optional, for progress bars during training

No GPU? You're fine. No GPU cluster? You're very fine. This is a ~1M parameter model. It's meant to run on anything.

---

## Setup

```bash
mkdir mini-transformer && cd mini-transformer
python -m venv venv
source venv/bin/activate           # Unix
venv\Scripts\activate              # Windows

pip install torch numpy matplotlib tqdm
```

Download the training data:

```bash
# The classic tinyshakespeare dataset (~1MB)
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Verify
wc -c input.txt     # should be ~1,115,000 bytes
```

Project layout:

```
mini-transformer/
├── input.txt           ← Shakespeare training data
├── model.py            ← the architecture
├── train.py            ← training loop
├── sample.py           ← generation from a trained checkpoint
├── notebook.ipynb      ← optional: experiments and plots
├── checkpoints/        ← saved models
└── README.md
```

---

## Requirements

### Must have

#### 1. Character-level tokenization

```python
text = open('input.txt').read()
chars = sorted(list(set(text)))             # unique characters
vocab_size = len(chars)                      # ~65 for Shakespeare
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)
```

Yes, it's that simple. Character-level is bad for production but perfect for understanding.

#### 2. Data loader

Split into train/val (90/10). At each training step, sample a batch of random subsequences of length `block_size` (the context window for this model). For each subsequence of length `block_size`, the target is the same sequence shifted by one character — each position predicts the character that follows.

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x, y
```

#### 3. The model — every piece from scratch

You must implement each of these as a PyTorch `nn.Module`. **No `nn.MultiheadAttention`** and no `nn.TransformerBlock`. Those are what you're building.

##### Single-head self-attention

```python
class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v        # (B, T, head_size)
        return out
```

Read Lesson 5 again if this is confusing. Every line corresponds to a step in the math.

##### Multi-head attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

##### Feed-forward MLP

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),     # expand
            nn.ReLU(),                          # activation
            nn.Linear(4 * n_embd, n_embd),     # compress back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

##### Transformer block (attention + MLP + residuals + norms)

```python
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # pre-norm + residual
        x = x + self.ffwd(self.ln2(x))   # pre-norm + residual
        return x
```

Notice the **pre-norm** pattern — layer norm is applied *before* the sub-block, not after (that's the original paper's "post-norm"). Pre-norm is what modern models use because it trains more stably.

##### The full model

```python
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                              # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                             # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                          # (B, T, vocab_size)

        if targets is None:
            return logits, None

        # Compute loss
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]              # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature      # take the last timestep
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
```

#### 4. Training loop

- **Optimizer:** AdamW with learning rate ~3e-4
- **Batches:** 64 sequences × 256 characters (or smaller if memory is tight)
- **Steps:** 5000-10000 (≈ 15-30 minutes on a laptop CPU for the small config)
- **Evaluate every 500 steps**: print train loss and val loss
- **Save** the model and loss history to a checkpoint when done

```python
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_losses, val_losses = [], []

for step in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        # Estimate loss on both splits
        with torch.no_grad():
            model.eval()
            train_l = estimate_loss("train")
            val_l = estimate_loss("val")
            train_losses.append((step, train_l))
            val_losses.append((step, val_l))
            print(f"step {step}: train loss {train_l:.4f}, val loss {val_l:.4f}")
            model.train()
```

#### 5. Generation

After training, generate a ~500-character sample starting from a newline token:

```python
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
```

Save the sample to `samples/sample_step_final.txt`.

#### 6. Loss curve plot

Save a matplotlib plot of training and validation loss over steps. It should show a clear downward trend with train and val loss tracking each other (no wild overfitting on 1MB of Shakespeare).

### Stretch goals (pick ≥2 — the real learning happens here)

- **Compare model sizes.** Train three configurations: tiny (2 layers, 2 heads, 32 embd), small (4 layers, 4 heads, 64 embd), medium (6 layers, 6 heads, 128 embd). Plot samples side-by-side. How much does each level of capacity buy you?
- **Temperature experiments.** After training, generate samples at `temperature ∈ {0.2, 0.5, 0.8, 1.0, 1.3}`. How does the output change?
- **Top-k vs top-p.** Implement top-p sampling yourself. Generate with both. Compare.
- **RoPE instead of learned positions.** Replace the learned position embedding with RoPE (Lesson 3). Train a new model. Does it generate better Shakespeare? (It probably won't for such a small model — but you'll deeply understand RoPE after implementing it.)
- **Attention visualization.** After training, run a single forward pass on a short prompt and plot the attention weights (`wei` from the Head class) as a heatmap. You'll literally see which tokens attend to which.
- **Train on something else.** Swap Shakespeare for Linux kernel source, Python code, your own journal, Chinese poetry — anything with a clear style. The model will learn it.
- **Parameter count.** Print the total parameter count and compare to larger models: your tiny model has ~1M parameters; GPT-2 Small has 124M; Llama 3 8B has 8B.

---

## Evaluation rubric — how to know you're done

- [ ] Model is implemented from scratch — no `nn.MultiheadAttention` or `nn.TransformerEncoderLayer`
- [ ] Training loop runs to completion without errors
- [ ] Final train loss < 1.5 and val loss < 1.6 (on the default config)
- [ ] Generated sample "looks like Shakespeare" at first glance — characters form English-ish words, punctuation is plausible, there are capitalized speaker names followed by colons
- [ ] Loss curve plot saved and shows monotonic decrease (with normal noise)
- [ ] Code is commented — each sub-module has a docstring explaining what it does
- [ ] You can explain, without looking at code, what each line of `Head.forward` does and why
- [ ] You completed at least 2 stretch goals
- [ ] README includes your final loss numbers, sample output, and any observations

---

## Common pitfalls

- **Skipping the Karpathy video.** You'll waste time reinventing intuition that's explained clearly in the video. Watch it first, once, without pausing. Then build.
- **Forgetting `masked_fill` for the causal mask.** Without it, your model "cheats" by seeing the future, training loss drops suspiciously fast, and generation is garbage. Add the mask and re-check.
- **Using `nn.MultiheadAttention`.** You are not allowed to. The entire point is to implement it. If you import it, you're cheating yourself of the understanding.
- **Forgetting to shift targets by 1.** At position `i`, the target should be the token at position `i+1`, not the token at position `i`. Get this wrong and your model learns to be a copy machine.
- **Wrong loss shape.** `F.cross_entropy` expects `(N, V)` logits and `(N,)` targets. You need to reshape from `(B, T, V)` to `(B*T, V)`. A common silent bug: you train but the loss is way off.
- **`block_size` too large for CPU.** 256 is fine; 512 starts to get slow on a CPU. If training is painfully slow, drop block_size to 128 and batch_size to 32.
- **Not cropping generation context.** Your `generate` function must crop the context to `block_size` before each step, or you'll exceed the trained context length and the position embeddings will break.
- **Validation loss much higher than training loss.** Means you're overfitting. Add dropout (0.2 is typical), or train fewer steps, or use a larger val split.
- **Learning rate too high.** If train loss explodes to NaN, your learning rate is too high. Try `1e-4` instead of `3e-4`.
- **Reading the wrong checkpoint.** If you add `sample.py`, make sure it loads the final checkpoint, not an intermediate one.
- **"My samples are garbage."** At step 500, samples will be garbage — random characters. At step 3000, they'll start having vaguely word-shaped segments. At step 8000, they'll look Shakespearean at a glance. If you're still seeing random characters at step 5000, something is wrong — check the causal mask, check the target shift, check the loss is actually decreasing.

---

## Hyperparameters (starting point)

```python
# Small config — trains on laptop CPU in ~20 min
block_size   = 256    # context window (characters)
batch_size   = 64     # sequences per training step
n_embd       = 384    # hidden dimension
n_head       = 6      # attention heads
n_layer      = 6      # transformer blocks
dropout      = 0.2
learning_rate = 3e-4
max_iters    = 5000
eval_interval = 500
eval_iters   = 200   # number of batches for loss estimation

# Tiny config — trains in ~5 min, for quick iteration
# block_size=128, batch_size=32, n_embd=128, n_head=4, n_layer=4, max_iters=3000
```

---

## Cost estimate

**Free.** Runs entirely on your machine. If you want to train larger configs faster, a Google Colab free T4 GPU is plenty — but you don't need it for the required work.

---

## What to deliver

```
mini-transformer/
├── input.txt
├── model.py            ← the TinyGPT architecture
├── train.py            ← training loop (runnable)
├── sample.py           ← load a checkpoint and generate (runnable)
├── checkpoints/
│   └── tinygpt_final.pt
├── samples/
│   ├── sample_step_500.txt
│   ├── sample_step_2000.txt
│   └── sample_final.txt   ← at least the final one
├── plots/
│   └── loss_curve.png
└── README.md
```

README should include:
- Final train/val loss
- Your sample output (~500 chars) pasted verbatim
- What you experimented with (which stretch goals)
- 2-3 things you learned that surprised you
- The total parameter count of your model

---

## Going further (after you finish)

- Re-read **[Karpathy's nanoGPT repo](https://github.com/karpathy/nanoGPT)**. It's a ~300-line production-quality version of what you just built. Compare your code to his.
- Try **[Anthropic's Transformer Circuits Thread](https://transformer-circuits.pub/)**. Now that you know how a transformer works, their interpretability research will make much more sense.
- Implement the **Chinchilla scaling laws** — pick a compute budget, compute the optimal model size and training tokens, train to that point. You'll directly experience scaling.
- Move to a **subword tokenizer** (tiktoken) and train on a larger dataset (OpenWebText, MiniPile). Measure the quality jump.
- Read the **original transformer paper** (*Attention Is All You Need*). Now that you've implemented it, every equation will feel obvious.

---

## References

- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*. https://www.youtube.com/watch?v=kCc8FmEb1nY (watch this first)
- Karpathy, *nanoGPT* — production-quality reference. https://github.com/karpathy/nanoGPT
- Vaswani et al. (2017), *Attention Is All You Need*. https://arxiv.org/abs/1706.03762
- Jay Alammar, *The Illustrated Transformer*. https://jalammar.github.io/illustrated-transformer/
- poloclub, *Transformer Explainer* (interactive). https://poloclub.github.io/transformer-explainer/
- PyTorch docs, *Tensor operations you'll use*. https://pytorch.org/docs/stable/torch.html
- Anthropic, *A Mathematical Framework for Transformer Circuits*. https://transformer-circuits.pub/2021/framework/index.html

---

[← Previous Project](02-token-economics.md) | [Back to LLM Fundamentals](../README.md)
