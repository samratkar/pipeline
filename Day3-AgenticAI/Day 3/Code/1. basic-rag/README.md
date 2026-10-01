# Lecture RAG System

A local Retrieval-Augmented Generation (RAG) system for querying PDF lecture notes.

## Stack

| Component | Technology |
|-----------|-----------|
| PDF loading | PyMuPDF (open source) |
| Text splitting | LangChain (open source) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (open source, local) |
| LLM | OpenAI `gpt-4o-mini` |
| Web UI | Streamlit (open source) |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your OpenAI API key
```

### 3. Ingest documents

```bash
python ingest.py
```

This reads every PDF in `documents/`, chunks the text, generates embeddings via OpenAI, and stores them locally in `chroma_db/`. Run this once (or again whenever you add new documents).

### 4. Query

**Web UI (recommended):**
```bash
streamlit run app.py
```

**CLI — interactive:**
```bash
python query.py
```

**CLI — one-shot:**
```bash
python query.py "What is the backpropagation algorithm?"
```

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI secret key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for answers |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `RETRIEVAL_K` | `5` | Number of chunks retrieved per query |

## File structure

```
RAG/
├── documents/          ← put your PDFs here
├── chroma_db/          ← auto-created after ingest
├── config.py           ← centralised settings
├── ingest.py           ← document → vector store pipeline
├── rag.py              ← core RAG chain (shared)
├── query.py            ← CLI interface
├── app.py              ← Streamlit web UI
├── requirements.txt
├── .env.example
└── .env                ← your secrets (git-ignored)
```
