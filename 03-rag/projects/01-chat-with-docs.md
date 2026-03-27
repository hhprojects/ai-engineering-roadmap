# Chat with Your Docs

🟢 **Beginner**

The "Hello World" of RAG. Load your own files, embed them, and ask questions in natural language. This is the foundation everything else builds on.

## What You'll Build

A CLI app that loads a folder of markdown/text files, chunks them, creates embeddings, stores them in ChromaDB, and lets you ask natural language questions. The LLM answers based on your documents, not its training data.

## What You'll Learn

- Text chunking strategies (fixed-size and recursive)
- Creating and using embeddings
- Vector database operations with ChromaDB
- Building a basic RAG pipeline end-to-end
- Prompt engineering for grounded generation

## Tech Stack

- Python 3.11+
- ChromaDB
- `fastembed` (free, local) or OpenAI `text-embedding-3-small`
- `openai` or `anthropic` SDK for generation
- `click` or `typer` for CLI

## Requirements

- Load all `.md` and `.txt` files from a specified directory
- Implement at least 2 chunking strategies (fixed-size with overlap, recursive by headers)
- Embed chunks and store in a local ChromaDB collection
- Accept natural language queries via CLI
- Retrieve top-k relevant chunks (default k=5)
- Pass retrieved chunks + query to an LLM with a grounding prompt
- Show the answer AND the source chunks it was based on
- Handle the case where no relevant documents are found
- Support re-indexing when documents change
- Persist the ChromaDB collection between runs

## Stretch Goals

- Add a conversation mode with follow-up questions (maintain chat history)
- Compare answer quality between chunking strategies on the same questions
- Add metadata filtering (e.g., only search files from a specific folder)

## Hints

- `fastembed` gives you free, local embeddings — no API key needed for the embedding step
- Your grounding prompt matters a lot: tell the LLM to only answer from the provided context and say "I don't know" if the context doesn't cover it
- Start with a small set of documents you know well — it's easier to evaluate quality when you know the right answers

## Cost Estimate

Free with `fastembed` + Groq. ~$0.10 with OpenAI embeddings.

---

[← Back to RAG](../README.md)
