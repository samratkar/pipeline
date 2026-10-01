import argparse
import os
from datetime import datetime

from langfuse import Evaluation

import config
from observability import flush, get_langfuse
from orchestrator import run_saga
from prompts import get_agent_prompt

# -----------------------------------
# Experiment Tracking (Langfuse datasets + experiment runs)
# -----------------------------------
#
# Runs the whole Saga over every paper in a Langfuse dataset and stores
# the result as a dataset run. Compare runs (models, prompt versions,
# app versions) side by side under Datasets -> research-papers.
#
#   python experiment.py
#   python experiment.py --model gpt-4o-mini
#   python experiment.py --run-name "review-prompt-v2" --no-ragas

DATASET = "research-papers"
PAPERS = ["paper.pdf", "paper1.pdf", "paper2.pdf"]


def ensure_dataset(lf):

    try:
        lf.get_dataset(DATASET)
    except Exception:
        lf.create_dataset(
            name=DATASET,
            description="Research papers for the Validation / Review / Summary saga"
        )

    for pdf in PAPERS:

        if not os.path.exists(pdf):
            continue

        # Stable id -> re-running upserts instead of duplicating items
        lf.create_dataset_item(
            dataset_name=DATASET,
            id=f"{DATASET}-{os.path.splitext(pdf)[0]}",
            input={"pdf": pdf}
        )

    return lf.get_dataset(DATASET)


def make_task(model, use_ragas, run_name):

    async def task(*, item, **kwargs):

        data = item.input if hasattr(item, "input") else item["input"]

        result = await run_saga(
            data["pdf"],
            model=model,
            use_ragas=use_ragas,
            session_id=run_name,
            push_scores=False
        )

        return {
            "status": result["status"],
            "error": result["error"],
            "validation": result["metadata"].get("validation"),
            "review": result["metadata"].get("review_comments"),
            "summary": result["metadata"].get("summary"),
            "scores": result["scores"],
        }

    return task


# Item level evaluator: publishes the saga's evaluation scores on each item
def saga_scores_evaluator(*, input, output, expected_output, metadata, **kwargs):

    return [
        Evaluation(name=name, value=s["value"], comment=s.get("comment"))
        for name, s in output["scores"].items()
    ]


# Run level evaluator: average of every metric across all papers
def average_scores_evaluator(*, item_results, **kwargs):

    totals = {}

    for item in item_results:
        for evaluation in item.evaluations:
            totals.setdefault(evaluation.name, []).append(evaluation.value)

    return [
        Evaluation(name=f"avg_{name}", value=sum(values) / len(values))
        for name, values in totals.items()
    ]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Override the model for all agents")
    parser.add_argument("--run-name")
    parser.add_argument("--no-ragas", action="store_true")
    args = parser.parse_args()

    lf = get_langfuse()

    if lf is None:
        raise SystemExit("Experiments need Langfuse - set LANGFUSE_* keys in .env")

    dataset = ensure_dataset(lf)

    model = args.model or config.AGENT_MODEL

    prompt_versions = {
        name: get_agent_prompt(name).label
        for name in ("paper-validation-agent", "paper-review-agent", "paper-summary-agent")
    }

    run_name = args.run_name or f"{model}-{datetime.now():%Y%m%d-%H%M%S}"

    result = dataset.run_experiment(
        name="paper-review-saga",
        run_name=run_name,
        description=f"Saga run with {model}",
        task=make_task(args.model, not args.no_ragas, run_name),
        evaluators=[saga_scores_evaluator],
        run_evaluators=[average_scores_evaluator],
        max_concurrency=1,      # saga_state.json / outputs are shared
        metadata={
            "model": model,
            "app_version": config.APP_VERSION,
            **prompt_versions
        }
    )

    print(result.format())

    flush()


if __name__ == "__main__":
    main()
