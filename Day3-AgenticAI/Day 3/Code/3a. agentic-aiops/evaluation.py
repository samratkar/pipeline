import asyncio

import config
from observability import observation

# -----------------------------------
# Agent Evaluation (RAGAS + custom evaluators)
# -----------------------------------
#
# Returns {metric_name: {"value": float, "comment": str}}. The orchestrator
# attaches every metric as a Langfuse score on the run's trace.

SUMMARY_INSTRUCTION = "Summarize this research paper in approximately 300 words."
REVIEW_INSTRUCTION = "Review this research paper: strengths, weaknesses and suggestions."

TARGET_SUMMARY_WORDS = 300
CONTEXT_CHUNK_CHARS = 1000


# -------- custom (deterministic) evaluators --------

def evaluate_validation(validation):

    return {
        "validation_valid": {
            "value": 1.0 if validation["verdict"] == "VALID" else 0.0,
            "comment": validation["verdict"]
        },
        "validation_confidence": {
            "value": float(validation["confidence"]),
            "comment": "Self-reported confidence of the Validation agent"
        },
    }


def evaluate_review(review):

    sections = ("strengths", "weaknesses", "suggestions")

    # A section counts as complete when it has at least 2 points
    complete = sum(1 for s in sections if len(review[s]) >= 2)

    return {
        "review_completeness": {
            "value": complete / len(sections),
            "comment": ", ".join(f"{s}={len(review[s])}" for s in sections)
        },
    }


def evaluate_summary_length(summary):

    words = len(summary.split())

    score = max(0.0, 1 - abs(words - TARGET_SUMMARY_WORDS) / TARGET_SUMMARY_WORDS)

    return {
        "summary_length_score": {
            "value": round(score, 3),
            "comment": f"{words} words (target {TARGET_SUMMARY_WORDS})"
        },
    }


# -------- RAGAS evaluators --------

def _chunk(text, size=CONTEXT_CHUNK_CHARS):

    return [text[i:i + size] for i in range(0, len(text), size)]


async def _safe(name, coro):

    try:
        result = await coro
        return name, {"value": float(result.value), "comment": result.reason}
    except Exception as e:
        print(f"RAGAS metric {name} failed: {e}")
        return name, None


async def evaluate_with_ragas(paper_excerpt, review_text, summary):

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness, SummaryScore

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    llm = llm_factory(config.EVAL_MODEL, client=client, max_tokens=4096)

    faithfulness = Faithfulness(llm=llm)
    summary_score = SummaryScore(llm=llm)

    contexts = _chunk(paper_excerpt)

    try:

        results = await asyncio.gather(

            # Is every claim in the summary supported by the paper text?
            _safe("summary_faithfulness", faithfulness.ascore(
                user_input=SUMMARY_INSTRUCTION,
                response=summary,
                retrieved_contexts=contexts
            )),

            # Does the summary capture the key information of the paper, concisely?
            _safe("summary_score", summary_score.ascore(
                reference_contexts=contexts,
                response=summary
            )),

            # Is the review grounded in the paper (no hallucinated claims)?
            _safe("review_faithfulness", faithfulness.ascore(
                user_input=REVIEW_INSTRUCTION,
                response=review_text,
                retrieved_contexts=contexts
            )),
        )

    finally:
        await client.close()

    return {name: value for name, value in results if value is not None}


# -------- entry point --------

async def evaluate_run(paper_excerpt, metadata, use_ragas=True):

    with observation("evaluation", as_type="evaluator") as span:

        scores = {}

        scores.update(evaluate_validation(metadata["validation_result"]))
        scores.update(evaluate_review(metadata["review_result"]))
        scores.update(evaluate_summary_length(metadata["summary"]))

        if use_ragas:
            scores.update(await evaluate_with_ragas(
                paper_excerpt,
                metadata["review_comments"],
                metadata["summary"]
            ))

        if span:
            span.update(output={k: v["value"] for k, v in scores.items()})

    return scores
