# Build a Character-Level Transformer

🟠 **Advanced**

Nothing beats building it yourself. Implement a transformer from scratch, train it on Shakespeare, and watch it learn to generate text one character at a time.

## What You'll Build

A character-level language model built from scratch in PyTorch, following Andrej Karpathy's approach. You'll implement attention, embeddings, and the full training loop — then train it on ~1MB of Shakespeare text using just your CPU.

## What You'll Learn

- Transformer architecture from the ground up (not just using APIs)
- Self-attention mechanism implementation
- Embedding layers and positional encoding
- Training loops, loss functions, and optimization
- How model size affects quality (experiment with layer count)

## Tech Stack

- Python 3.11+
- PyTorch (CPU-only is fine)
- NumPy
- Matplotlib (for loss curves)

## Requirements

- Implement character-level tokenization (char → int mapping)
- Build a self-attention head from scratch (no `nn.MultiheadAttention`)
- Implement multi-head attention by combining multiple heads
- Build the full transformer block: attention → layer norm → feed-forward → layer norm
- Stack multiple blocks into a model
- Implement the training loop with AdamW optimizer
- Train on Shakespeare's complete works (~1MB)
- Generate sample text after training
- Plot training loss over time
- Model should train in under 30 minutes on CPU
- Include comments explaining each component

## Stretch Goals

- Experiment with different model sizes (vary heads, layers, embedding dim) and compare sample quality
- Add a temperature parameter to generation and show how it affects output
- Implement top-k sampling alongside greedy and temperature-based generation

## Hints

- Start with [Karpathy's video](https://www.youtube.com/watch?v=kCc8FmEb1nY) — watch it once, then build without pausing
- The Shakespeare dataset is available at `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Keep the model small (4 layers, 4 heads, 64-dim embeddings) — you want fast iteration, not GPT-4

---

[← Back to LLM Fundamentals](../README.md)
