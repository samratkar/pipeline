"""
RAG evaluation with RAGAS, linked to Langfuse.

For each question below, this script runs it through the live RAG chain
(rag.ask), scores the resulting (question, retrieved_contexts, answer) with
RAGAS reference-free metrics, and pushes each metric as a score onto that
question's Langfuse trace — so evaluation results and traces live side by
side in the Langfuse UI.

Run:
    python evaluate.py
"""

import asyncio
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
import rag

# Questions live in eval.json — edit that file to add questions relevant to
# your own lecture content.
EVAL_QUESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.json")

with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
    EVAL_QUESTIONS = json.load(f)["eval_questions"]

# Reference-free metrics: no ground-truth answers needed, just question +
# retrieved contexts + generated answer.
METRICS = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextPrecisionWithoutReference(),
]


def init_metrics():
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=config.OPENAI_MODEL, openai_api_key=config.OPENAI_API_KEY, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY)
    )
    run_config = RunConfig()
    for metric in METRICS:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embeddings
        metric.init(run_config)


async def score_answer(question: str, contexts: list, answer: str) -> dict:
    sample = SingleTurnSample(user_input=question, retrieved_contexts=contexts, response=answer)
    scores = {}
    for metric in METRICS:
        scores[metric.name] = await metric.single_turn_ascore(sample)
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

    print(f"\nEvaluating {len(EVAL_QUESTIONS)} question(s)...\n")
    totals = {m.name: [] for m in METRICS}

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        result = rag.ask(chain, question, session_id="ragas-evaluation")
        scores = await score_answer(question, result["contexts"], result["answer"])

        print(f"[{i}] {question}")
        for name, value in scores.items():
            print(f"    {name}: {value:.3f}")
            totals[name].append(value)
            rag.score_trace(result["trace_id"], f"ragas_{name}", value)
        print()

    print("--- Averages across all questions ---")
    for name, values in totals.items():
        if values:
            print(f"{name}: {sum(values) / len(values):.3f}")

    rag.flush()
    if config.LANGFUSE_ENABLED:
        print(f"\nScores pushed to Langfuse — view them under each trace at {config.LANGFUSE_HOST}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
