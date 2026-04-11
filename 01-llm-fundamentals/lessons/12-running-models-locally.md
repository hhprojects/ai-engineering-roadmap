# Lesson 12 — Running Models Locally

> **The single sentence version:** You can run surprisingly capable language models on a laptop, a gaming GPU, or a modest server using open-weight models and quantization — and knowing how is a superpower for privacy, cost control, offline use, and really *understanding* what the models do.

So far, every time you've talked to a language model in this course, it's been someone else's model running on someone else's hardware. This chapter flips that. We'll cover why you'd want to run a model locally, what tools make it trivial (Ollama, llama.cpp, vLLM), what quantization actually does, and what you can realistically run on the hardware you already own.

---

## Why run locally?

Four real reasons. You need at least one of them, otherwise just keep using the APIs.

1. **Privacy and data sovereignty.** Some data can't leave your machine. Medical records, customer PII under GDPR, proprietary source code, confidential business documents. If you can't upload it to a third party, you have to run inference somewhere you control.
2. **Cost at scale.** API costs scale linearly with usage. A self-hosted model has a fixed infrastructure cost and near-zero marginal cost per request. The break-even point depends on model size and usage volume, but for many workloads (high-volume embedding, bulk summarization, heavy evals), self-hosting is genuinely cheaper above some threshold.
3. **Offline or edge.** Mobile apps, embedded devices, airline pilots, field researchers, submarines. Sometimes the network isn't there.
4. **Learning.** Running a model yourself — watching it load, seeing how much RAM it consumes, tweaking the sampling parameters — builds intuition that API calls never will. For a course like this one, it's worth doing at least once.

What you *shouldn't* run locally (most of the time): customer-facing production traffic that expects frontier-model quality. The gap between open-weight models and the closed frontier (Claude Opus, GPT-5, Gemini 3.1 Pro) remains real for many tasks, especially coding and complex reasoning. Use local for the workloads above, not as a cost-cut on your flagship product.

---

## The parameters and the hardware

The first question for any local inference setup: **can my hardware even run this model?** Two constraints:

- **VRAM (GPU) or unified memory (Apple Silicon)** determines whether the model fits at all. A 7B model at 16-bit precision is ~14GB; at 4-bit it's ~4GB.
- **Memory bandwidth** determines how fast it runs. Apple Silicon (unified memory at 200-800 GB/s) trades off: large capacity, decent bandwidth, no swapping. Consumer NVIDIA GPUs (dedicated VRAM at 700-1000 GB/s) are faster but have less capacity. Datacenter GPUs (H100, H200) are fastest but priced out of reach.

Rough rule of thumb for memory cost in GB, by parameter count and precision:

| Params | FP16 | Q8 | Q5 | Q4 | Q3 |
|---:|---:|---:|---:|---:|---:|
| 1B | 2 | 1 | 0.7 | 0.5 | 0.4 |
| 3B | 6 | 3 | 2 | 1.5 | 1.2 |
| 7B | 14 | 7 | 5 | 4 | 3 |
| 13B | 26 | 13 | 9 | 7 | 5 |
| 34B | 68 | 34 | 22 | 18 | 14 |
| 70B | 140 | 70 | 45 | 40 | 30 |
| 405B | 810 | 405 | — | 230 | — |

Add ~20% on top for the KV cache at typical context lengths.

Practical consequences:

- **8GB VRAM** (RTX 3070, 3060 Ti) — can run 7B models at Q4, 3B at Q8
- **16GB VRAM** (RTX 4080, 4090 partial, RTX 5070 Ti) — can run 13B at Q5-Q8
- **24GB VRAM** (RTX 3090, 4090, 5090) — can run 34B at Q4, 13B comfortably at full precision
- **48-80GB VRAM** (A100, H100, dual 4090s) — can run 70B at Q4, 13B at full precision
- **Apple Silicon M3 Max 64GB** — can run 70B at Q4 or Q5 at reasonable speed
- **Apple Silicon M3 Ultra 192GB** — can run 405B at Q4 (slowly but it works)

Nothing consumer runs Llama 3.1 405B at full precision. Nothing consumer runs DeepSeek-R1 671B at full precision. But quantized versions of almost everything else are within reach.

---

## Quantization, explained

Models are trained in floating-point — usually 16-bit or 32-bit per parameter. **Quantization** compresses each parameter to fewer bits (8-bit, 5-bit, 4-bit, 3-bit, and even 2-bit) with a small quality loss. A 4-bit quantized 7B model has the same parameter *count* as a 16-bit 7B model, but takes only 1/4 the memory.

How does this work? The key trick is **group quantization**: the model's weights are divided into small groups (e.g., 128 values per group), each group has its own scale and zero-point, and within a group every weight is stored as a 4-bit integer that gets dequantized on the fly during inference. The scale and zero-point are themselves stored in higher precision (usually 16-bit). The result is that most of your memory is spent on the small integers, while the overhead of the scaling factors is amortized across the group.

The quality hit depends on the bit-width:

- **Q8 / 8-bit** — essentially indistinguishable from full precision. Almost always worth it. Roughly halves memory vs. FP16.
- **Q5 / 5-bit** — very minor quality degradation on most tasks. The sweet spot for most consumer inference.
- **Q4 / 4-bit** — noticeable but usually acceptable degradation. This is where the open-weight community mostly lives.
- **Q3 / 3-bit** — visible quality loss. Use only when you can't fit anything higher.
- **Q2 / 2-bit** — significant degradation, but sometimes the only way to fit a huge model on consumer hardware. Last resort.

Different quantization schemes exist — **GGUF** (the llama.cpp format, used everywhere in Ollama), **AWQ**, **GPTQ**, **EXL2** — and they differ in quality at the same bit-width. For most hobbyist use, GGUF via llama.cpp or Ollama is the default. For production serving, AWQ or GPTQ with vLLM is faster.

**Practical note:** you'll see model names like `llama-3.1-8b-instruct-Q4_K_M.gguf`. The `Q4_K_M` part means "4-bit, K-quant method, medium size" — a specific flavor of 4-bit quantization that's a good default. The `K_S` variants are smaller and slightly worse; the `K_L` variants are larger and slightly better. Most of the time, pick `K_M`.

---

## The tools you should know

### Ollama — the easy button

Start here unless you have a specific reason not to. Ollama is a single-binary installer that bundles llama.cpp with a nice CLI and a REST API.

```bash
curl -fsSL https://ollama.com/install.sh | sh        # Linux/macOS
# or download from ollama.com for Windows

ollama run llama3.2                                  # downloads and runs interactively
ollama run qwen2.5:7b                                # pick a specific model
ollama list                                          # see what you have
```

Once it's running, Ollama exposes an OpenAI-compatible API on `localhost:11434`. Your code can point at it and use the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"   # required by the SDK but not checked
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain tokenization in one paragraph."}]
)
```

This is the killer feature. Anything you wrote against the OpenAI API can be switched to a local model by changing `base_url` — no other code changes.

**What's in Ollama's library (early 2026):**

- **Llama 3.2 / 3.3 / 4** (Meta) — general-purpose, the most-pulled
- **Qwen 2.5 / 3** (Alibaba) — excellent multilingual, 0.5B–72B sizes, best small-model quality
- **Mistral / Mixtral** — solid general-purpose, strong on European languages
- **DeepSeek-R1** (1.5B–671B) — open reasoning models
- **Gemma 3** (Google, 270M–27B) — efficient and small
- **Phi** (Microsoft) — small and punching above their weight
- **Llava / Llama 3.2 Vision** — multimodal (image + text)
- **Nomic Embed / BGE / mxbai-embed** — embedding models
- **TinyLlama 1.1B** — runs on a potato, useful for testing pipelines

Ollama handles downloading, quantization format, GPU detection, and serving. For 90% of local use, this is enough.

### llama.cpp — the engine everyone uses

Ollama is llama.cpp with a friendly wrapper. For finer control — custom grammars, specific quantization variants, exotic hardware support (including bare-metal ARM, old CPUs, and embedded boards) — you go straight to llama.cpp.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

./main -m models/llama-3-8b.Q4_K_M.gguf -p "Hello, world" -n 128
./server -m models/llama-3-8b.Q4_K_M.gguf --port 8080   # OpenAI-compatible API
```

llama.cpp has CPU inference, Metal (Apple), CUDA (NVIDIA), ROCm (AMD), and Vulkan backends. It runs almost anywhere. It's also where most new quantization research lands first.

### vLLM — the production serving engine

When you need to *serve* a local model to many concurrent users with high throughput, vLLM is the tool. It uses paged KV cache management (PagedAttention), continuous batching, and FP8/FP16/AWQ/GPTQ backends to maximize utilization of your GPUs.

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000
```

Use vLLM when:
- You're serving more than a handful of concurrent requests
- You have a real GPU (NVIDIA, preferably datacenter-grade)
- Throughput matters more than single-request latency
- You're deploying to production, not iterating on a laptop

For development and single-user use, Ollama is easier. For production serving, vLLM is faster by a wide margin.

### MLX (Apple Silicon only)

Apple's own ML framework, optimized for unified memory. The `mlx-lm` package lets you run models at near-optimal speed on M-series chips, often faster than llama.cpp's Metal backend. Worth checking out if you're on a Mac and want the best performance.

### Hugging Face Transformers (for experimenting)

The `transformers` library is the one everyone uses for fine-tuning, research, and experimentation. It's not fast for serving — use vLLM or llama.cpp for that — but it's the most flexible for "I want to load this specific checkpoint and do something weird with it."

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

inputs = tokenizer("Hello, world", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

Use this when: you need to inspect hidden states, extract attention maps, load a fine-tuned checkpoint, run a model that isn't in GGUF format yet, or do research.

---

## The on-laptop experience, what to expect

A realistic picture of inference speed on common setups:

| Setup | Model | Speed (tokens/s) |
|---|---|---:|
| MacBook Pro M3 16GB | Llama 3.2 3B Q4 | 30-50 |
| MacBook Pro M3 Max 64GB | Llama 3.1 70B Q4 | 5-10 |
| RTX 4090 24GB | Llama 3.1 8B Q8 | 100+ |
| RTX 4090 24GB | Llama 3.1 70B Q4 | 15-20 |
| Pair of RTX 4090s (48GB) | Llama 3.1 70B Q8 | 25-30 |
| H100 80GB | Llama 3.1 70B FP16 | 80+ |
| H100 80GB with vLLM batching | Llama 3.1 70B FP16 | 1000+ aggregate |

Context: Claude and GPT APIs typically serve at 50-100 tokens/sec. Groq serves at 200-500 tokens/sec. So a good consumer setup matches API speeds for small models and is noticeably slower for large ones.

**The speed that matters most** is the "thinking speed" — how long until you see the first token. For interactive use, <2 seconds is fine, <1 is great, <0.5 is crisp. Local models on consumer hardware typically hit first-token latency in the 0.5-2 second range for prompts under a few thousand tokens, which is very usable.

---

## When to use local vs. API (decision table)

| Scenario | Recommendation |
|---|---|
| Prototyping a new idea | API (fast iteration, no setup) |
| Production customer-facing chat | API (quality matters, support matters) |
| Bulk background summarization at scale | Self-hosted (big cost win) |
| Sensitive data that can't leave premises | Self-hosted (required) |
| Offline mobile app | Self-hosted / on-device (required) |
| Learning how LLMs work | Self-hosted at least once (insight) |
| Code assistant on your laptop | Self-hosted with a capable open model (privacy + offline) |
| Research that needs model internals | Self-hosted via Transformers (required) |
| Evaluation / red-teaming | Mix — API for flagship comparisons, local for heavy batch |

---

## Common pitfalls

- **Expecting frontier quality from a 7B model.** Open-weight models have closed the gap dramatically, but at 7B you're still well below GPT-5 or Claude Opus on hard tasks. Set expectations accordingly — or step up to 70B+ and accept the hardware cost.
- **Not trying quantization.** You'll get much better quality from a Q5 70B than a full-precision 13B on equal hardware. Bigger-and-quantized beats smaller-and-full-precision for most tasks.
- **Assuming all 4-bit is the same.** AWQ, GPTQ, GGUF Q4_K_M, EXL2 4bpw — all "4-bit," all quite different in practice. Check benchmarks for the specific scheme.
- **Measuring speed without measuring throughput.** A consumer GPU gets ~30 tok/s for one request but collapses under concurrent load. vLLM and production serving engines only shine under concurrency.
- **Running a 13B model and thinking you've "replaced" your API usage.** Test the 13B model on your *actual* prompts and measure the quality gap. You'll usually find the local model is good enough for 60-80% of your traffic and not good enough for the rest. That's still a big win — just don't oversell it.
- **Forgetting about the tokenizer.** Llama uses a different tokenizer from GPT. Qwen uses a different tokenizer from both. The same prompt costs different amounts in different tokens. If you're comparing costs, compare at the same tokenization.

---

## What to remember from this lesson

- You can run capable models locally with Ollama (easy), llama.cpp (flexible), or vLLM (fast at scale).
- Quantization shrinks models 2-8× with modest quality loss. Q4_K_M is a good default; Q5 or Q8 if you have the memory.
- A modern laptop can run 7B-13B models comfortably; a gaming GPU can run 34B; a high-end workstation can run 70B quantized.
- Use local for privacy, offline, cost-at-scale, and learning. Use APIs for production quality, flagship capability, and ease.
- Ollama exposes an OpenAI-compatible API, so switching from API to local is usually a one-line change.

---

## References

- Ollama, *Model library*. https://ollama.com/library
- llama.cpp, *GitHub repository*. https://github.com/ggerganov/llama.cpp
- vLLM, *GitHub repository and docs*. https://github.com/vllm-project/vllm
- Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. https://arxiv.org/abs/2208.07339
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. https://arxiv.org/abs/2210.17323
- Lin et al., *AWQ: Activation-aware Weight Quantization*. https://arxiv.org/abs/2306.00978
- Hugging Face, *Transformers library*. https://huggingface.co/docs/transformers/index
- Apple, *MLX framework*. https://ml-explore.github.io/mlx/

---

[← Lesson 11](11-token-economics.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 13: Choosing a Model](13-choosing-a-model.md)
