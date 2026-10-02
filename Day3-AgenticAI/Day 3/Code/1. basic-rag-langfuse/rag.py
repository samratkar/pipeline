"""
Core RAG chain — reusable by both query.py (CLI) and app.py (Streamlit).

When Langfuse keys are configured (see config.LANGFUSE_ENABLED), every call to
ask() is traced end-to-end (retrieval + generation) in Langfuse, and the
returned trace_id can be used to attach scores — either user feedback or
RAGAS evaluation metrics (see evaluate.py) — back onto that trace.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVAL_K},
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


def ask(chain: RetrievalQA, question: str, session_id: str = None, user_id: str = None) -> dict:
    """
    Returns:
        {
            "answer": str,
            "sources": [{"file": str, "page": int, "snippet": str}, ...],
            "contexts": [str, ...],   # raw retrieved chunk text (for RAGAS)
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
        }

    result = chain.invoke({"query": question}, config=invoke_config)
    answer = result["result"]

    sources = []
    contexts = []
    seen = set()
    for doc in result.get("source_documents", []):
        meta = doc.metadata
        contexts.append(doc.page_content)
        key = (meta.get("source_file", ""), meta.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": meta.get("source_file", meta.get("source", "")),
                "page": meta.get("page", "?"),
                "snippet": doc.page_content[:300].strip(),
            })

    return {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
        "trace_id": handler.last_trace_id if handler else None,
    }


def score_trace(trace_id: str, name: str, value, comment: str = None):
    """Attach a score (user feedback, a RAGAS metric, ...) to a Langfuse trace."""
    if not (config.LANGFUSE_ENABLED and trace_id):
        return
    langfuse_client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)


def flush():
    """Force-send any buffered Langfuse events. Call at the end of short-lived scripts."""
    if langfuse_client:
        langfuse_client.flush()
