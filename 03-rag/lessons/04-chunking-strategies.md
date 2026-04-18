# 04 — Chunking Strategies

> Chunking is how you cut documents into the retrievable units the rest of the pipeline depends on; it is the single biggest pre-retrieval quality lever, and most teams get it wrong before they even look at embeddings.

If retrieval is broken, chunking is the first place to look — before changing the embedding model, before adding reranking, before blaming the LLM. A chunk that is too small is semantically naked; a chunk that is too large is a haystack around the needle; a chunk that splits in the middle of a table destroys the information the user was looking for. This lesson covers every chunking strategy worth knowing in 2026 and tells you which one to reach for first.

## Why chunking exists

There are three reasons you do not embed whole documents:

1. **Embedding models have a token limit.** Most range from 512 to 8192 tokens per input. A 200-page PDF does not fit.
2. **A single vector compresses a lot of signal.** Embedding a 40-page document into 1536 floats loses the fine-grained topic structure. Retrieval becomes coarse — "this document might be relevant" — instead of "this paragraph is the answer."
3. **The LLM has a context budget.** Even with 1M token windows, you want to pass the model the most relevant snippets, not 40 pages of possibly-relevant prose. Smaller chunks let you pass more distinct sources.

The goal of chunking is to produce units that are each **coherent** (one topic, one idea, one answer) and **independently meaningful** (readable in isolation). Everything else is a trade-off between these two goals.

## The five chunking strategies you will actually use

### 1. Fixed-size chunking

Split the text into blocks of `N` tokens (or characters), optionally with an overlap of `M` tokens between adjacent chunks. That is it.

```python
def fixed_chunks(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    tokens = text.split()  # in real code, use a tokenizer
    chunks = []
    for i in range(0, len(tokens), size - overlap):
        chunk = " ".join(tokens[i:i + size])
        if chunk:
            chunks.append(chunk)
    return chunks
```

**When to use it:** your first implementation, always. It is 10 lines of code, it handles any input, and it gives you a baseline to measure every other strategy against.

**When it hurts you:** it slices mid-sentence, splits tables in half, and ignores document structure. A chunk that starts with "which is why" is useless without the previous sentence.

**Good defaults:** 500–800 tokens, 10–15% overlap. The Anthropic contextual retrieval blog used 800 tokens. OpenAI's cookbook examples use 500. Start with 800 and measure.

### 2. Recursive character splitting

LangChain's `RecursiveCharacterTextSplitter` and LlamaIndex's `SentenceSplitter` implement the same idea: try to split on paragraph breaks first, then sentence breaks, then word breaks, and only fall back to mid-word splits as a last resort. The priority list is usually `["\n\n", "\n", ". ", " ", ""]`.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document)
```

**When to use it:** your *second* implementation, always. It is the sensible default for plain prose — Markdown, HTML-stripped web pages, plain text — and it fixes the worst failures of fixed-size chunking with no additional cost.

**When it hurts you:** structured documents where the structure matters. A PDF with tables, a Python file, a Jupyter notebook — all of these need a format-aware splitter.

### 3. Document-structure-aware chunking

Parse the document's actual structure and chunk along structural boundaries:

- **Markdown:** chunk by headings. `##` sections become chunks; subsections can be merged upward if they are too small.
- **HTML:** chunk by `<h2>` / `<h3>` / `<section>` / `<article>`.
- **PDFs:** chunk by section (`pdfplumber` or `pymupdf` gives you layout information) and **extract tables separately** — do not try to chunk a table as if it were prose.
- **Code:** chunk by function, class, or module (`tree-sitter` gives you the AST).
- **Jupyter notebooks:** one chunk per code cell + its following markdown.

This is where you stop losing information. A markdown-structured chunker preserves the heading hierarchy; a table chunker keeps rows and headers together. The extra engineering cost is repaid the first time a user asks a question whose answer is literally "row 3 of table 2 on page 14."

**Practical tip:** store the heading path as metadata. `{"section": "Chapter 3 > Reranking > Cross-encoders"}` becomes both a filter and a breadcrumb you can show the user.

### 4. Semantic chunking

Instead of fixed boundaries, embed every sentence, then group adjacent sentences whose embeddings are close and break where the embedding distance spikes. The chunks end up at natural topic shifts rather than at arbitrary token counts.

```python
# Pseudocode
sentences = split_into_sentences(text)
sentence_vecs = embed_each(sentences)
breakpoints = [i for i in range(1, len(sentences))
               if cosine_distance(sentence_vecs[i], sentence_vecs[i-1]) > threshold]
chunks = group_by_breakpoints(sentences, breakpoints)
```

**When to use it:** long-form narrative text where topics shift at irregular intervals — essays, blog posts, long memos, interview transcripts.

**When it hurts you:** it is expensive (an embedding call per sentence at index time), the threshold is another thing to tune, and on structured documents it loses to a format-aware splitter. In 2026 most practitioners have quietly stopped using it — contextual chunking (below) gives more reliable quality for less complexity.

### 5. Contextual (LLM-augmented) chunking

This is the 2024 state-of-the-art from Anthropic. Instead of changing *where* you split, you change *what* you index. For each chunk, ask a cheap LLM to write a short blurb situating the chunk within the full document, and prepend that blurb to the chunk before embedding.

```
You are given a document and a chunk from it. Write a short (50–100 token)
context that would help someone understand the chunk if they read it alone.
Return only the context, nothing else.

<document>...</document>
<chunk>...</chunk>
```

A chunk that was originally "The company's revenue grew 3% over the previous quarter" becomes:

```
[Acme Corp Q3 2025 earnings; this sentence reports quarterly revenue growth
compared to Q2 2025.] The company's revenue grew 3% over the previous quarter.
```

The Anthropic experiments showed this alone reduced retrieval failure by 35%, and combined with contextual BM25 and reranking cut failures by 67%. With Claude Haiku 4.5 and prompt caching, the cost is roughly **$1 per million document tokens** — trivial for most corpora.

**When to use it:** any corpus large enough that long context is not an option and where chunks lose important context on their own (reports, emails, multi-entity documents). Contextual retrieval gets its own dedicated lesson (lesson 09) because it is a large enough topic to deserve one.

## Overlap and why you need it

Fixed-size and recursive splitters should always have some overlap — typically 10–20% of the chunk size. The reason is that any hard boundary will eventually fall in the middle of a relevant answer, and overlap gives the retriever a second chance at that answer by duplicating the boundary text in both chunks.

There are two costs:

- **Index size grows** proportional to `1 / (1 - overlap_ratio)`. A 15% overlap means a 17% bigger index.
- **Retrieved chunks can be near-duplicates.** A question whose answer sits in the overlap region pulls both adjacent chunks, and the LLM sees the same sentence twice. Deduplicate at retrieval time (match on prefix or Jaccard similarity) or reduce top-k.

Do not use large overlaps to "just be safe." 100 tokens on an 800-token chunk is plenty.

## The small-to-big pattern

This is the single most important chunking idea after contextual retrieval: **the chunk you embed is not the chunk you pass to the LLM.**

Embed small units (one sentence, one paragraph) so retrieval is precise. But when a small unit is retrieved, expand it to a larger context window — the surrounding paragraph, the whole section, the full document — before sending it to the generator. The generator gets rich context; the retriever gets precise matching.

Variants:

- **Sentence-window retrieval (LlamaIndex):** embed single sentences, retrieve them, then return the sentence plus its `k` neighbours as context.
- **Parent-document retriever (LangChain):** chunk small (say 200 tokens) for embedding, but store each small chunk with a pointer to a larger parent chunk (say 2000 tokens). Retrieve small, return parent.
- **Hierarchical chunking:** index chunks at multiple granularities — sentence, paragraph, section — and let the retriever or a router pick which level to serve per query.

Small-to-big alone usually improves answer faithfulness noticeably and costs almost nothing. It is the second thing to build after a baseline.

## Chunk metadata: do not skip this

For every chunk, store at least:

- `document_id` — the source document's primary key.
- `chunk_index` — position within the document, useful for retrieving neighbours.
- `source_path` or `source_url` — where it came from.
- `heading_path` — the document structure (e.g., `"Chapter 3 > Chunking"`).
- `page_number` — for PDFs.
- `created_at` — ingestion time, for freshness filtering.
- `doc_type` — `"pdf"`, `"markdown"`, `"html"`, `"code"`, etc.

You will not know in advance which of these you will need. Store all of them. Disk is free; reindexing is not.

## Practical chunking recipe

If I were starting a new RAG project today, I would do this in order:

1. **Ingest with format-aware parsers.** PDFs through `pymupdf` or `pdfplumber` (with table extraction); HTML through `beautifulsoup4`; markdown through a structural splitter; code through `tree-sitter`. Do not homogenise everything into plain text — you are throwing away information.
2. **Chunk with `RecursiveCharacterTextSplitter`** at 800 tokens / 100 overlap as a baseline. Preserve heading paths as metadata.
3. **Tables get their own chunks.** One table, one chunk (or one row per chunk if the table is huge). Include column headers in each chunk. Never splice a table with surrounding prose.
4. **Build an eval set immediately.** 30–50 questions whose correct chunks you can identify. This is lesson 05, and it is the only way to judge whether chunking changes are helping or hurting.
5. **Add contextual chunking** (lesson 09) when you have an eval set showing that naked chunks are failing retrieval. Do not add it speculatively.
6. **Add small-to-big retrieval** when the LLM is producing faithful answers but missing context around the retrieved snippet.

The full pipeline rarely needs more than this. Semantic chunking, agentic chunking, and exotic strategies are almost never the lever that matters.

## What to remember

- Chunking is the single highest-leverage lever before retrieval. Measure before and after every chunking change.
- Start with recursive character splitting at 800 tokens / 100 overlap. It is a genuinely good default.
- Document structure matters. Use a format-aware splitter and extract tables separately.
- Store heading path, page number, and doc type as metadata. You will thank yourself later.
- Small-to-big retrieval — embed small, return large — is an easy win on top of any chunker.
- Contextual chunking (lesson 09) is the current state of the art for losing less information per chunk.
- Overlap is necessary but 10–15% is plenty. More is waste.

## References

- Pinecone, *Chunking strategies for LLM applications*. https://www.pinecone.io/learn/chunking-strategies/
- Anthropic, *Introducing Contextual Retrieval*. https://www.anthropic.com/news/contextual-retrieval
- LangChain, *RecursiveCharacterTextSplitter*. https://python.langchain.com/docs/how_to/recursive_text_splitter/
- LlamaIndex, *Production RAG — decoupling retrieval from synthesis and small-to-big*. https://developers.llamaindex.ai/python/framework/optimizing/production_rag/
- LlamaIndex, *Sentence window retriever*. https://developers.llamaindex.ai/
- `pdfplumber` for table-aware PDF extraction. https://github.com/jsvine/pdfplumber
- `pymupdf` for fast PDF parsing. https://pymupdf.readthedocs.io/
