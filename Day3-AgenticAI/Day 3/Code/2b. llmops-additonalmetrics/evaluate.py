"""
RAG evaluation with RAGAS, linked to Langfuse.

For each question in eval.json, this script runs it through the live RAG chain
(rag.ask), scores the resulting (question, retrieved_contexts, answer) with
RAGAS metrics, and pushes each metric as a score onto that question's Langfuse
trace — so evaluation results and traces live side by side in the Langfuse UI.

Two kinds of metrics:
  * Reference-free (always run): need only question + contexts + answer.
  * Reference-based (run only for questions that have a "reference" answer in
    eval.json): compare the generated answer against a ground-truth answer.

Each run gets its own Langfuse session, and the per-metric averages are also
pushed as session-level scores, so whole runs can be compared in Langfuse.

Run:
    python evaluate.py
"""

import asyncio
import json
import math
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AspectCritic,
    ContextRelevance,
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseGroundedness,
    ResponseRelevancy,
    SemanticSimilarity,
)
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
import rag

# Questions live in eval.json — edit that file to add questions relevant to
# your own lecture content. Each entry is either a plain question string or
# {"question": "...", "reference": "..."}; a non-empty reference enables the
# reference-based metrics for that question.
EVAL_QUESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.json")

with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
    EVAL_QUESTIONS = [
        q if isinstance(q, dict) else {"question": q}
        for q in json.load(f)["eval_questions"]
    ]

# Reference-free metrics: no ground-truth answers needed, just question +
# retrieved contexts + generated answer.
METRICS = [
    Faithfulness(),                           # are the answer's claims supported by the contexts?
    ResponseRelevancy(),                      # does the answer address the question?
    LLMContextPrecisionWithoutReference(),    # are the useful chunks ranked first?
    ContextRelevance(),                       # are the retrieved chunks relevant to the question?
    ResponseGroundedness(),                   # how grounded is the answer overall in the contexts?
    AspectCritic(                             # binary LLM-judge check (1 = pass)
        name="conciseness",
        definition="Does the response answer the question concisely, without "
                   "unnecessary repetition or content unrelated to the question?",
    ),
]

# Reference-based metrics: only run when a question has a "reference" answer.
REFERENCE_METRICS = [
    LLMContextRecall(),       # did retrieval find everything the reference answer needs?
    FactualCorrectness(),     # claim-level overlap between answer and reference
    SemanticSimilarity(),     # embedding similarity between answer and reference
]


def init_metrics():
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=config.OPENAI_MODEL, openai_api_key=config.OPENAI_API_KEY, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY)
    )
    run_config = RunConfig()
    for metric in METRICS + REFERENCE_METRICS:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embeddings
        metric.init(run_config)


async def score_answer(question: str, contexts: list, answer: str, reference: str = None) -> dict:
    sample = SingleTurnSample(
        user_input=question, retrieved_contexts=contexts, response=answer, reference=reference or None
    )
    metrics = METRICS + (REFERENCE_METRICS if reference else [])
    scores = {}
    for metric in metrics:
        try:
            value = await metric.single_turn_ascore(sample)
        except Exception as e:
            print(f"    [{metric.name} failed] {e}")
            continue
        # Langfuse rejects NaN; RAGAS returns it when a metric can't be computed.
        if value is not None and not math.isnan(value):
            scores[metric.name] = float(value)
    return scores


async def main_async():
    if not os.path.exists(config.CHROMA_DB_DIR):
        print("Vector store not found. Run 'python ingest.py' first.")
        sys.exit(1)

    if not config.LANGFUSE_ENABLED:
        print(
            "Langfuse keys not set — scores will only be printed, not pushed.\n"
            "Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env to enable.\n"
        )

    print("Loading RAG chain...")
    vector_store = rag.load_vector_store()
    chain = rag.build_rag_chain(vector_store)

    print("Initializing RAGAS metrics...")
    init_metrics()

    # One Langfuse session per run, so runs can be compared side by side.
    session_id = f"ragas-evaluation-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"\nEvaluating {len(EVAL_QUESTIONS)} question(s) — session '{session_id}'...\n")
    totals = {}

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        question = item["question"]
        result = rag.ask(chain, question, session_id=session_id)
        scores = await score_answer(
            question, result["contexts"], result["answer"], item.get("reference")
        )

        print(f"[{i}] {question}")
        for name, value in result["metrics"].items():
            print(f"    {name}: {value}")
            if isinstance(value, (int, float)):
                totals.setdefault(name, []).append(value)
        for name, value in scores.items():
            print(f"    ragas_{name}: {value:.3f}")
            totals.setdefault(f"ragas_{name}", []).append(value)
            rag.score_trace(result["trace_id"], f"ragas_{name}", value)
        print()

    print("--- Averages across all questions ---")
    for name, values in totals.items():
        avg = sum(values) / len(values)
        print(f"{name}: {avg:.3f}  (n={len(values)})")
        rag.score_session(session_id, f"avg_{name}", avg, comment=f"mean over {len(values)} question(s)")

    rag.flush()
    if config.LANGFUSE_ENABLED:
        print(
            f"\nScores pushed to Langfuse — per-question scores on each trace, run averages "
            f"on session '{session_id}' at {config.LANGFUSE_HOST}"
        )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
