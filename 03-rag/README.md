# 3 — RAG (Retrieval-Augmented Generation)

**Goal:** Build retrieval pipelines from basic to production-grade with evaluation.

## Learning Objectives

- Implement end-to-end RAG pipelines (chunking → embedding → retrieval → generation)
- Compare vector search, keyword search, and hybrid approaches
- Evaluate retrieval and generation quality systematically

---

## 📚 Readings & Resources

| Resource | Type | Why |
|----------|------|-----|
| [freeCodeCamp: Learn RAG from Scratch](https://www.freecodecamp.org/news/mastering-rag-from-scratch/) | Tutorial | Comprehensive free tutorial by a LangChain engineer |
| [Pinecone: What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/) | Guide | Clear conceptual overview with diagrams |
| [Pinecone: Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/) | Guide | Essential reading on text splitting approaches |
| [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | GitHub | Massive collection of RAG technique notebooks — gold mine |
| [Jason Liu: Levels of RAG Complexity](https://jxnl.co/writing/2024/02/28/levels-of-complexity-rag-applications/) | Blog | Mental model for RAG maturity levels |
| [DeepLearning.AI: RAG Course](https://learn.deeplearning.ai/courses/retrieval-augmented-generation/) | 🎬 Course | Free short course covering the full RAG pipeline |
| [Complete RAG Tutorial 2026](https://www.youtube.com/watch?v=vT-DpLvf29Q) | 🎬 YouTube | Comprehensive crash course with code |

---

## 🔨 Projects

| # | Project | Difficulty | Description |
|---|---------|------------|-------------|
| 1 | [Chat with Your Docs](projects/01-chat-with-docs.md) | 🟢 Beginner | Load files, embed, query with natural language |
| 2 | [Hybrid Search Engine](projects/02-hybrid-search.md) | 🟡 Intermediate | Vector + BM25 with reranking and a web UI |
| 3 | [Multi-Source RAG with Eval](projects/03-multi-source-eval.md) | 🟠 Advanced | PDFs, web, CSV — unified index with evaluation harness |

---

## Key Concepts

After completing this section, you should understand:

- What embeddings are and how they represent semantic meaning
- Chunking strategies: fixed-size, recursive, semantic
- Vector similarity search (cosine, dot product)
- Why naive RAG fails and what "lost in the middle" means
- Hybrid search: combining vector and keyword (BM25) approaches
- Reranking and why it improves retrieval quality
- Evaluation metrics: hit rate, MRR, faithfulness, relevance
- When RAG is the right approach vs. fine-tuning or long-context models

---

[← Prompt Engineering](../02-prompt-engineering/) | [Home](../README.md) | [Next → Agents](../04-agents/)
