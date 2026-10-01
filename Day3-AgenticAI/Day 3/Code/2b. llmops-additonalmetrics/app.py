"""
Streamlit web UI for the RAG system.

Run with:
    streamlit run app.py
"""

import asyncio
import os
import threading
import uuid
import streamlit as st

st.set_page_config(
    page_title="Live Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Live Q&A Chatbot on the topic 'API-Driven Cloud Native Solutions'")
st.caption(f"Based on Lecture notes by Prof. Shreyas Rao ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})")

# ── Load chain once per session ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector store…")
def get_chain():
    if not os.path.exists("chroma_db"):
        return None
    import rag
    vector_store = rag.load_vector_store()
    return rag.build_rag_chain(vector_store)


chain = get_chain()

if chain is None:
    st.error(
        "Vector store not found. "
        "Run `python ingest.py` in your terminal first, then refresh this page."
    )
    st.stop()

# ── RAGAS live evaluation ────────────────────────────────────────────────────
# Every question asked here is scored the same way evaluate.py scores its
# fixed question list — run in a background thread so the answer renders
# immediately and scores land on the trace in Langfuse a few seconds later.
@st.cache_resource(show_spinner="Initializing RAGAS metrics…")
def get_ragas_evaluator():
    import config
    if not config.LANGFUSE_ENABLED:
        return None
    import evaluate as ragas_eval
    ragas_eval.init_metrics()
    return ragas_eval


ragas_eval = get_ragas_evaluator()


def score_in_background(ragas_eval, rag_module, question, result):
    try:
        scores = asyncio.run(
            ragas_eval.score_answer(question, result["contexts"], result["answer"])
        )
        for name, value in scores.items():
            rag_module.score_trace(result["trace_id"], f"ragas_{name}", value)
        rag_module.flush()
    except Exception as e:
        print(f"[live RAGAS scoring failed] {e}")

# ── Chat history ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    # Groups every turn of this browser session into one Langfuse session.
    st.session_state.session_id = str(uuid.uuid4())
if "feedback_sent" not in st.session_state:
    st.session_state.feedback_sent = set()


def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for src in sources:
            score = f" · relevance {src['score']:.2f}" if src.get("score") is not None else ""
            st.markdown(
                f"**{src['file']}** — Page {src['page']}{score}\n\n"
                f"> {src['snippet']}…"
            )


def render_feedback(trace_id):
    # Rendered from history (not only right after answering) so the click,
    # which triggers a rerun, is still captured and sent to Langfuse.
    if not trace_id:
        return
    feedback = st.feedback("thumbs", key=f"fb_{trace_id}")
    if feedback is not None and trace_id not in st.session_state.feedback_sent:
        import rag as rag_module
        rag_module.score_trace(
            trace_id,
            "user_feedback",
            feedback,
            comment="Streamlit thumbs feedback",
            data_type="BOOLEAN",
        )
        rag_module.flush()
        st.session_state.feedback_sent.add(trace_id)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))
        render_feedback(msg.get("trace_id"))

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about the lectures…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            import rag as rag_module
            result = rag_module.ask(chain, question, session_id=st.session_state.session_id)

        if ragas_eval and result["trace_id"]:
            threading.Thread(
                target=score_in_background,
                args=(ragas_eval, rag_module, question, result),
                daemon=True,
            ).start()

        st.markdown(result["answer"])
        render_sources(result["sources"])
        render_feedback(result["trace_id"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "trace_id": result["trace_id"],
    })
