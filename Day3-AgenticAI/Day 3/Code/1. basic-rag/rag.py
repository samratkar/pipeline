"""
Core RAG chain — reusable by both query.py (CLI) and app.py (Streamlit).
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import config

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
        temperature=0, #hyper-parameter to render proper facts
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVAL_K},
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", #concatenate all retrieved documents together
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


def ask(chain: RetrievalQA, question: str) -> dict:
    """
    Returns:
        {
            "answer": str,
            "sources": [{"file": str, "page": int, "snippet": str}, ...]
        }
    """
    result = chain.invoke({"query": question})
    answer = result["result"]

    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        meta = doc.metadata
        key = (meta.get("source_file", ""), meta.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": meta.get("source_file", meta.get("source", "")),
                "page": meta.get("page", "?"),
                "snippet": doc.page_content[:300].strip(),
            })

    return {"answer": answer, "sources": sources}
