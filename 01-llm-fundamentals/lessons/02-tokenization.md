# Lesson 2 — Tokenization

> **The single sentence version:** Tokens are the units your model actually sees — not words, not characters, but subword chunks chosen by a learned algorithm that balances vocabulary size against sequence length.

If Lesson 1 convinced you that LLMs are next-token predictors, your next question should be: *what's a token?* This chapter answers that, and shows you why the answer has real consequences for your bill, your context window, and how well your model handles other languages.

---

## Why not just use words?

The naïve approach is word-level tokenization: split text on whitespace and punctuation, assign every unique word an integer ID.

```python
["Don't", "you", "love", "🤗", "Transformers", "?"]
```

Problems:

1. **Vocabulary explosion.** Every inflection is a new token: `love`, `loves`, `loved`, `loving`, `lovingly`. English has ~170,000 common words and many more rare ones. Your embedding matrix (Lesson 3) has one row per token — a 170k vocabulary with 4096-dim embeddings is ~700M parameters just for the vocabulary.
2. **Unknown words.** Anything not in the training vocabulary becomes `<UNK>`. The model literally cannot represent it.
3. **Doesn't work for most of the world.** Chinese and Japanese don't use spaces. Turkish and Finnish agglutinate — you can form entirely new "words" at will.

Character-level tokenization — one token per character — fixes the unknown-word problem and has a tiny vocabulary, but pays for it with sequences that are 5-10× longer, which makes training and inference proportionally more expensive. Karpathy's mini-transformer (Project 3) uses character-level tokenization precisely because it's simple and avoids the vocabulary problem — but no serious production model does.

---

## Subword tokenization — the actual answer

Modern LLMs use **subword tokenization**. The vocabulary contains both whole common words *and* smaller fragments that can combine to form rare ones. The word `annoyingly` might become `["annoying", "ly"]` or `["annoy", "ing", "ly"]`, depending on the tokenizer.

This gives you three properties at once:

- **Small vocabulary** (typically 30k-200k tokens)
- **No unknown words** — anything can be reconstructed from smaller fragments, ultimately down to bytes
- **Good compression** — common words are one token, so common text stays short

There are three algorithms you need to know by name.

### Byte Pair Encoding (BPE)

The most popular. Used by **GPT, Llama, Gemma, Qwen, Mistral**, and most open-weight models.

BPE starts with a base vocabulary (characters or bytes) and repeatedly merges the most frequent adjacent pair until the vocabulary reaches a target size. Walking through a toy example:

```
Start:   ("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4)
                                                                  base: [b, g, h, n, p, u]
```

Most frequent pair is `u g` (appears 15 times across `hug` and `pug`). Merge:

```
After:   ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4)
                                                                  vocab: [..., ug]
```

Next most frequent is `u n` (16 times). Merge:

```
After:   ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4)
                                                                  vocab: [..., ug, un]
```

And so on, for tens of thousands of merges. GPT-2 learned 50,000 merges on top of a 256-byte base vocabulary for a final vocabulary of 50,257 tokens.

**Byte-level BPE** (used by GPT-2, GPT-4, most modern models) starts from the 256 possible bytes instead of Unicode characters, which guarantees that any input — including emoji, Chinese characters, or garbled bytes — can be tokenized without `<UNK>` tokens.

### WordPiece

Used by **BERT, DistilBERT, Electra** and most encoder models from the Google lineage.

Very similar to BPE — iterative bottom-up merging — but instead of picking the most *frequent* pair, WordPiece picks the pair that maximizes the likelihood of the training data. Concretely, it scores each candidate merge by:

```
score(a, b) = freq(ab) / (freq(a) × freq(b))
```

This favors pairs that appear together *more often than chance predicts*. Two letters that often co-occur get merged; two letters that just happen to both be common (like `e` and space) don't. It's a more informative merge criterion than raw frequency.

### Unigram + SentencePiece

Used by **T5, ALBERT, XLNet, mBART, and any model trained with SentencePiece's unigram mode**.

Unigram works *top-down* instead of bottom-up. It starts with a large candidate vocabulary, scores how much each token contributes to representing the training data, and removes the least useful tokens until it hits the target vocabulary size. This is probabilistic — a word can be tokenized multiple ways during training, which acts as a regularizer.

**SentencePiece** is a library (not an algorithm) that runs BPE or Unigram directly on raw text, including spaces. It encodes whitespace as a special character `▁` ("lower one-eighth block"), which is why you'll see `▁hello` in Llama tokenizer outputs. This makes it work cleanly on languages that don't delimit words with spaces.

---

## What this means for your API bill

Tokens are not words, and different models count them differently. A few facts worth internalizing:

- **In English, ~1 token ≈ 0.75 words.** A 1000-word essay is roughly 1300 tokens.
- **Spaces usually belong to the following word.** `" the"` is one token, not `" "` + `"the"`.
- **Punctuation is usually its own token.** `.`, `,`, `?` each cost one token.
- **Emoji and non-Latin scripts are expensive.** A single Chinese character may be 2-3 tokens in a GPT tokenizer. An emoji can be 4+ tokens. An Arabic or Thai sentence can easily cost 2× what the same English sentence costs.
- **JSON and code are surprisingly expensive.** Curly braces, colons, and whitespace all consume tokens. A deeply indented JSON document can spend 20% of its tokens on whitespace.
- **Same string, different token counts across providers.** OpenAI uses `tiktoken` (a fast BPE implementation); Claude uses a different BPE; Llama uses SentencePiece BPE; Gemini uses yet another. The same sentence can cost 15% more or fewer tokens depending on which tokenizer you're billed against. Your Project 2 — the token economics calculator — exists to make this visible.

---

## A working example

Let's tokenize this sentence in three ways:

> "Don't you love 🤗 Transformers?"

**GPT-4 (tiktoken cl100k_base):**
```
["Don", "'t", " you", " love", " ", "🤗", " Transform", "ers", "?"]    ~9 tokens
```

**Llama 3 (SentencePiece BPE):**
```
["▁Don", "'", "t", "▁you", "▁love", "▁", "🤗", "▁Transformers", "?"]   ~9 tokens
```

**Character-level (what your mini-transformer uses):**
```
["D", "o", "n", "'", "t", " ", "y", "o", ...]                            ~30 tokens
```

Same sentence, very different sequence lengths. The choice of tokenizer affects training compute, inference cost, context window utilization, and how "naturally" your model handles rare text.

---

## Common pitfalls

- **Assuming word count ≈ token count.** It's close for English prose, wildly wrong for code, JSON, or multilingual text. Always count tokens before assuming you fit in a context window.
- **Forgetting system prompts count.** Your 8k context window has to hold *your* prompt *and* the system prompt *and* the model's response. A 3000-token system prompt leaves you 5000 tokens to work with.
- **Copy-pasting between providers.** A prompt that fits in GPT-4's 128k context might not fit the same way in Claude's 200k context if the text happens to tokenize heavier in one than the other.
- **Measuring cost in words.** Always measure cost in tokens. Words are a lie your tokenizer tells you.

---

## What to remember from this lesson

- Modern LLMs tokenize at the subword level, not word or character.
- The main algorithms are BPE (GPT, Llama, Qwen), WordPiece (BERT), and Unigram (T5). SentencePiece is the library that implements BPE/Unigram on raw text.
- One English word ≈ 1.3 tokens on average; non-English text, code, and JSON cost more.
- Every provider uses a different tokenizer, so the same prompt costs different amounts across providers.
- Sequence length drives training cost, inference cost, and context window consumption. Tokens are the unit that matters.

Next chapter: what actually happens to those tokens once the model reads them.

---

## References

- Hugging Face Transformers, *Tokenizer summary*. https://huggingface.co/docs/transformers/tokenizer_summary
- Sennrich, Haddow, Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* (the BPE paper). https://huggingface.co/papers/1508.07909
- Kudo (2018), *Subword Regularization: Improving NMT Models with Multiple Subword Candidates* (Unigram). https://huggingface.co/papers/1804.10959
- Kudo & Richardson (2018), *SentencePiece: A simple and language independent subword tokenizer*. https://huggingface.co/papers/1808.06226
- OpenAI, *tiktoken* (the BPE tokenizer for GPT models). https://github.com/openai/tiktoken

---

[← Lesson 1](01-what-is-a-language-model.md) | [Back to LLM Fundamentals](../README.md) | [Next → Lesson 3: Embeddings](03-embeddings-and-positional-encoding.md)
