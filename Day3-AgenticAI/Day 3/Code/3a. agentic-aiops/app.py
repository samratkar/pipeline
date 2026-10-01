import asyncio
import json
import os
import uuid

import streamlit as st

import config
from observability import flush, score
from orchestrator import run_saga

# -----------------------------------
# Application / Demo UI (Streamlit)
#   streamlit run app.py
# -----------------------------------

UPLOAD_DIR = os.path.join(config.OUTPUT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODELS = ["gpt-5-nano", "gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini"]

st.set_page_config(page_title="Research Paper Agents", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ---------------- Sidebar ----------------

with st.sidebar:

    st.header("Run settings")

    default_model = config.AGENT_MODEL if config.AGENT_MODEL in MODELS else MODELS[0]
    model = st.selectbox("Agent model", MODELS, index=MODELS.index(default_model))

    evaluate = st.toggle("Run evaluation", value=True)
    use_ragas = st.toggle("Include RAGAS metrics", value=True, disabled=not evaluate)
    simulate_failure = st.toggle("Simulate Summary failure (rollback demo)", value=False)

    st.divider()
    st.caption(f"Langfuse tracing: {'on' if config.LANGFUSE_ENABLED else 'off'}")
    st.caption(f"Prompt label: {config.PROMPT_LABEL}")
    st.caption(f"App version: {config.APP_VERSION}")
    st.caption(f"Session: {st.session_state.session_id[:8]}")

# ---------------- Main ----------------

st.title("Research Paper Review Agents")
st.caption("Validation → Review → Summary, orchestrated as a Saga with compensation")

uploaded = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded and st.button("Run agents", type="primary"):

    pdf_path = os.path.join(UPLOAD_DIR, uploaded.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner("Running Validation, Review and Summary agents..."):

        st.session_state.result = asyncio.run(run_saga(
            pdf_path,
            model=model,
            evaluate=evaluate,
            use_ragas=use_ragas,
            simulate_failure=simulate_failure,
            session_id=st.session_state.session_id
        ))

        flush()

result = st.session_state.get("result")

if result:

    if result["status"] == "COMPLETED":
        st.success("Saga completed")
    else:
        st.error(f"Saga {result['status']}: {result['error']}")

    if result.get("trace_url"):
        st.markdown(f"[Open trace in Langfuse]({result['trace_url']})")

    st.caption(
        "Models: " + ", ".join(result["models"]) + "  |  Prompts: "
        + ", ".join(result["prompts"].values())
    )

    metadata = result["metadata"]

    tabs = st.tabs(["Validation", "Review", "Summary", "Evaluation", "Saga state"])

    with tabs[0]:
        st.markdown(metadata.get("validation", "_Rolled back / not produced_"))

    with tabs[1]:
        st.markdown(metadata.get("review_comments", "_Rolled back / not produced_"))

    with tabs[2]:
        st.markdown(metadata.get("summary", "_Rolled back / not produced_"))

    with tabs[3]:

        scores = result["scores"]

        if scores:

            cols = st.columns(4)

            for i, (name, s) in enumerate(scores.items()):
                cols[i % 4].metric(name, f"{s['value']:.2f}", help=s.get("comment"))

        else:
            st.info("No evaluation for this run")

    with tabs[4]:
        st.code(json.dumps(
            {k: v for k, v in result.items() if k != "metadata"},
            indent=2
        ), language="json")

    # User feedback -> Langfuse score on this trace
    st.divider()
    st.write("Was this output helpful?")

    feedback = st.feedback("thumbs", key=f"feedback-{result.get('trace_id')}")

    if feedback is not None and st.session_state.get("feedback_sent") != result.get("trace_id"):

        score(result.get("trace_id"), "user_feedback", float(feedback),
              data_type="BOOLEAN")
        flush()

        st.session_state.feedback_sent = result.get("trace_id")
        st.toast("Feedback sent to Langfuse")
