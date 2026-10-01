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

## Metrics in Langfuse

When Langfuse keys are set, each query becomes a trace. Langfuse records **latency, token usage and cost** from the LangChain callback on its own. On top of that, these scores are attached:

| Score | Type | Source | Meaning |
|-------|------|--------|---------|
| `retrieval_top_score` / `retrieval_mean_score` | Numeric | every query | Cosine similarity between the question and the best / average retrieved chunk |
| `unique_sources` | Numeric | every query | Number of distinct lecture PDFs among retrieved chunks |
| `answer_length_words` | Numeric | every query | Answer length |
| `refused_to_answer` | Boolean | every query | Answer said the information is not in the context |
| `top_source` | Categorical | every query | Lecture the best-matching chunk came from |
| `user_feedback` | Boolean | Streamlit thumbs | 1 = 👍, 0 = 👎 |
| `ragas_faithfulness` | Numeric | RAGAS | Answer claims supported by the retrieved context |
| `ragas_answer_relevancy` | Numeric | RAGAS | Answer addresses the question |
| `ragas_llm_context_precision_without_reference` | Numeric | RAGAS | Useful chunks ranked first |
| `ragas_nv_context_relevance` | Numeric | RAGAS | Retrieved chunks relevant to the question |
| `ragas_nv_response_groundedness` | Numeric | RAGAS | Answer grounded in the retrieved chunks |
| `ragas_conciseness` | Numeric (0/1) | RAGAS | LLM judge: answer is concise and on-topic |
| `ragas_context_recall` | Numeric | RAGAS, needs `reference` | Retrieval found what the reference answer needs |
| `ragas_factual_correctness` | Numeric | RAGAS, needs `reference` | Claim overlap between answer and reference |
| `ragas_semantic_similarity` | Numeric | RAGAS, needs `reference` | Embedding similarity between answer and reference |

RAGAS metrics run in the background for every question asked in the Streamlit app and for every question in `evaluate.py`. The reference-based metrics run only in `evaluate.py`, and only for questions in `eval.json` that have a non-empty `"reference"` answer.

Each `evaluate.py` run gets its own session (`ragas-evaluation-<timestamp>`). The run's averages are attached to that session as `avg_<metric>` scores, so you can compare runs, e.g. after changing `RETRIEVAL_K`. Every trace also stores `llm_model`, `embedding_model`, `retrieval_k` and `chunk_size` in its metadata for filtering.

To chart any of these, add a widget in Langfuse under **Dashboards** and pick the score name.

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
