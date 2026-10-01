"""
Core RAG chain — reusable by both query.py (CLI) and app.py (Streamlit).

When Langfuse keys are configured (see config.LANGFUSE_ENABLED), every call to
ask() is traced end-to-end (retrieval + generation) in Langfuse, and the
returned trace_id can be used to attach scores — either user feedback or
RAGAS evaluation metrics (see evaluate.py) — back onto that trace.

On top of that, ask() computes a handful of cheap, deterministic metrics
(retrieval similarity, source diversity, answer length, refusal detection)
and attaches them to the same trace, so every query has scores in Langfuse
even when no LLM-judged evaluation runs. Latency, token usage and cost are
captured by Langfuse automatically from the LangChain callback.
"""

import re
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config

if config.LANGFUSE_ENABLED:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    langfuse_client = get_client()
else:
    langfuse_client = None

_PROMPT_TEMPLATE = """You are a helpful academic assistant. Use the following
lecture excerpts to answer the question. If the answer is not in the context,
say so clearly — do not make up information.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_PROMPT_TEMPLATE,
)

# Phrases the model uses when it follows the prompt's "say so clearly" rule.
_REFUSAL_PATTERN = re.compile(
    r"not (in|within|covered in|mentioned in|found in|present in|provided in) the (context|excerpts|lecture)"
    r"|(context|excerpts) (does|do) not (contain|mention|provide|include)"
    r"|i (don't|do not) (know|have enough information)",
    re.IGNORECASE,
)


class ScoredRetriever(BaseRetriever):
    """Similarity retriever that keeps each chunk's cosine similarity to the
    question in doc.metadata["relevance_score"], so it shows up in the Langfuse
    retriever span and can be turned into retrieval-quality scores."""

    vector_store: Chroma
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = []
        for doc, distance in self.vector_store.similarity_search_with_score(query, k=self.k):
            # Chroma's default space returns squared L2 distance; OpenAI
            # embeddings are unit-length, so cosine similarity = 1 - d/2.
            doc.metadata["relevance_score"] = round(1 - distance / 2, 4)
            docs.append(doc)
        return docs


def load_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
    return Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings,
    )


def build_rag_chain(vector_store: Chroma) -> RetrievalQA:
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0,
    )
    retriever = ScoredRetriever(vector_store=vector_store, k=config.RETRIEVAL_K)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


def compute_heuristic_metrics(answer: str, source_documents: List[Document]) -> dict:
    """
    Cheap, reference-free metrics that need no extra LLM calls.

    Returns {name: (value, data_type)} where data_type is one of Langfuse's
    score types: NUMERIC, BOOLEAN or CATEGORICAL.
    """
    relevance = [d.metadata.get("relevance_score") for d in source_documents]
    relevance = [r for r in relevance if r is not None]
    files = [d.metadata.get("source_file", d.metadata.get("source", "")) for d in source_documents]

    metrics = {
        "answer_length_words": (len(answer.split()), "NUMERIC"),
        "unique_sources": (len(set(files)), "NUMERIC"),
        "refused_to_answer": (1 if _REFUSAL_PATTERN.search(answer) else 0, "BOOLEAN"),
    }
    if relevance:
        metrics["retrieval_top_score"] = (max(relevance), "NUMERIC")
        metrics["retrieval_mean_score"] = (round(sum(relevance) / len(relevance), 4), "NUMERIC")
    if files:
        # Which lecture the best-matching chunk came from — lets you see in
        # Langfuse which lectures are actually answering students' questions.
        metrics["top_source"] = (files[0], "CATEGORICAL")
    return metrics


def ask(chain: RetrievalQA, question: str, session_id: str = None, user_id: str = None) -> dict:
    """
    Returns:
        {
            "answer": str,
            "sources": [{"file": str, "page": int, "snippet": str, "score": float}, ...],
            "contexts": [str, ...],   # raw retrieved chunk text (for RAGAS)
            "metrics": {name: value}, # heuristic metrics (also pushed to Langfuse)
            "trace_id": str | None,   # Langfuse trace id (for scoring), if enabled
        }
    """
    invoke_config = {}
    handler = None
    if config.LANGFUSE_ENABLED:
        handler = CallbackHandler()
        invoke_config["callbacks"] = [handler]
        invoke_config["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_user_id": user_id,
            "langfuse_tags": ["basic-rag"],
            # Recorded on the trace so runs with different settings can be
            # filtered and compared in Langfuse.
            "llm_model": config.OPENAI_MODEL,
            "embedding_model": config.OPENAI_EMBEDDING_MODEL,
            "retrieval_k": config.RETRIEVAL_K,
            "chunk_size": config.CHUNK_SIZE,
        }

    result = chain.invoke({"query": question}, config=invoke_config)
    answer = result["result"]
    source_documents = result.get("source_documents", [])

    sources = []
    contexts = []
    seen = set()
    for doc in source_documents:
        meta = doc.metadata
        contexts.append(doc.page_content)
        key = (meta.get("source_file", ""), meta.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": meta.get("source_file", meta.get("source", "")),
                "page": meta.get("page", "?"),
                "snippet": doc.page_content[:300].strip(),
                "score": meta.get("relevance_score"),
            })

    trace_id = handler.last_trace_id if handler else None
    metrics = compute_heuristic_metrics(answer, source_documents)
    for name, (value, data_type) in metrics.items():
        score_trace(trace_id, name, value, data_type=data_type)

    return {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
        "metrics": {name: value for name, (value, _) in metrics.items()},
        "trace_id": trace_id,
    }


def score_trace(trace_id: str, name: str, value, comment: str = None, data_type: str = None):
    """Attach a score (user feedback, a RAGAS metric, ...) to a Langfuse trace.

    data_type is NUMERIC (default), BOOLEAN (0/1) or CATEGORICAL (string)."""
    if not (config.LANGFUSE_ENABLED and trace_id):
        return
    langfuse_client.create_score(
        trace_id=trace_id, name=name, value=value, comment=comment, data_type=data_type
    )


def score_session(session_id: str, name: str, value, comment: str = None):
    """Attach a score to a whole Langfuse session (e.g. an evaluation run's average)."""
    if not (config.LANGFUSE_ENABLED and session_id):
        return
    langfuse_client.create_score(session_id=session_id, name=name, value=value, comment=comment)


def flush():
    """Force-send any buffered Langfuse events. Call at the end of short-lived scripts."""
    if langfuse_client:
        langfuse_client.flush()
