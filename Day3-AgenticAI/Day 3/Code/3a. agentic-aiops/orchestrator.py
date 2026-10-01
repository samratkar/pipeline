import argparse
import asyncio
import json
import os
import uuid

from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunConfig,
    Runner,
)

import config
from document import read_pdf
from evaluation import evaluate_run
from observability import (
    current_trace_id,
    flush,
    init_observability,
    observation,
    score,
    trace_attributes,
    trace_url,
)
from paper_agents import build_agents, format_review, format_validation

# -----------------------------------
# Setup
# -----------------------------------

init_observability()

STATE_FILE = config.STATE_FILE
OUTPUT_DIR = config.OUTPUT_DIR

# -----------------------------------
# State Store
# -----------------------------------

def save_state(state):

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def load_state():

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)

    return {
        "status": "NOT_STARTED",
        "completed_agents": [],
        "metadata": {}
    }

# -----------------------------------
# Utility Functions
# -----------------------------------

def write_output(filename, content):

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created: {path}")

    return path


def delete_output(filename):

    path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")


def remove_agent_state(agent_name, metadata_keys, state):

    if agent_name in state["completed_agents"]:
        state["completed_agents"].remove(agent_name)

    for key in metadata_keys:
        state["metadata"].pop(key, None)

    save_state(state)


def describe_error(e):

    if isinstance(e, InputGuardrailTripwireTriggered):
        result = e.guardrail_result
        return f"Input guardrail '{result.guardrail.get_name()}' tripped: {result.output.output_info}"

    if isinstance(e, OutputGuardrailTripwireTriggered):
        result = e.guardrail_result
        return f"Output guardrail '{result.guardrail.get_name()}' tripped: {result.output.output_info}"

    return f"{type(e).__name__}: {e}"

# -----------------------------------
# Agents (Saga steps backed by OpenAI Agents SDK agents)
# -----------------------------------

class SagaAgent:

    name = None
    output_file = None
    metadata_keys = []

    def __init__(self, agent, prompt):

        self.agent = agent
        self.prompt = prompt

    async def run_agent(self, paper_text):

        # Links the LLM generations of this agent to its Langfuse prompt version
        with trace_attributes(prompt=self.prompt.client):

            result = await Runner.run(
                self.agent,
                paper_text,
                run_config=RunConfig(
                    workflow_name=self.name,
                    trace_metadata={
                        "prompt": self.prompt.label,
                        "model": str(self.agent.model)
                    }
                )
            )

        return result.final_output

    def compensate(self, state):

        print(f"Compensating {self.name}")

        with observation(f"compensate-{self.name}"):

            delete_output(self.output_file)

            remove_agent_state(
                self.name,
                self.metadata_keys,
                state
            )

# -----------------------------------

class ValidationAgent(SagaAgent):

    name = "ValidationAgent"
    output_file = "validation.txt"
    metadata_keys = ["validation", "validation_result"]

    async def execute(self, paper_text):

        print("\n[Validation Agent]")

        result = await self.run_agent(paper_text)

        text = format_validation(result)

        print(text)

        if result.verdict == "INVALID":
            raise Exception("Validation failed: " + result.justification)

        write_output(self.output_file, text)

        return {
            "validation": text,
            "validation_result": result.model_dump()
        }

# -----------------------------------

class ReviewAgent(SagaAgent):

    name = "ReviewAgent"
    output_file = "review.txt"
    metadata_keys = ["review_comments", "review_result"]

    async def execute(self, paper_text):

        print("\n[Review Agent]")

        result = await self.run_agent(paper_text)

        review = format_review(result)

        write_output(self.output_file, review)

        return {
            "review_comments": review,
            "review_result": result.model_dump()
        }

# -----------------------------------

class SummaryAgent(SagaAgent):

    name = "SummaryAgent"
    output_file = "summary.txt"
    metadata_keys = ["summary"]

    async def execute(self, paper_text):

        print("\n[Summary Agent]")

        # Simulate failure for testing rollback
        if "FAIL_SUMMARY" in paper_text:
            raise Exception("Simulated Summary Failure")

        summary = await self.run_agent(paper_text)

        write_output(self.output_file, summary)

        return {
            "summary": summary
        }

# -----------------------------------
# Saga Orchestrator
# -----------------------------------

class SagaOrchestrator:

    def __init__(self, model=None, session_id=None):

        self.model = model
        self.session_id = session_id or str(uuid.uuid4())

        self.completed = []

        self.state = {
            "status": "RUNNING",
            "completed_agents": [],
            "metadata": {}
        }

        save_state(self.state)

    def update_state(self, agent_name, metadata):

        self.state["completed_agents"].append(agent_name)

        self.state["metadata"].update(metadata)

        save_state(self.state)

        print(f"\nState Updated After {agent_name}")

    def rollback(self):

        print("\n================================")
        print("STARTING COMPENSATION")
        print("================================")

        with observation("saga-rollback", metadata={"agents": [a.name for a in self.completed]}):

            for agent in reversed(self.completed):

                agent.compensate(self.state)

        self.state["status"] = "ROLLED_BACK"

        save_state(self.state)

        print("\nFinal State:")

        print(json.dumps(self.state, indent=4))

        print("\n================================")
        print("ROLLBACK COMPLETE")
        print("================================")

    async def run(self, pdf_path, evaluate=True, use_ragas=True, simulate_failure=False,
                  push_scores=True):

        # Prompts are fetched per run so new Langfuse versions are picked up
        built = build_agents(self.model)

        validation = ValidationAgent(*built["ValidationAgent"])
        review = ReviewAgent(*built["ReviewAgent"])
        summary = SummaryAgent(*built["SummaryAgent"])

        steps = [validation, review, summary]

        models = sorted({str(step.agent.model) for step in steps})
        prompt_versions = {step.name: step.prompt.label for step in steps}

        result = {
            "pdf": str(pdf_path),
            "session_id": self.session_id,
            "models": models,
            "prompts": prompt_versions,
            "scores": {},
            "error": None,
        }

        # Trace level attributes used for filtering / comparing runs in Langfuse
        with trace_attributes(
            trace_name="paper-review-saga",
            session_id=self.session_id,
            version=config.APP_VERSION,
            tags=["saga", *models],
            metadata={
                "pdf": os.path.basename(str(pdf_path)),
                "models": ",".join(models),
                **{f"prompt_{k}": v for k, v in prompt_versions.items()}
            },
        ):

            with observation(
                "paper-review-saga",
                as_type="chain",
                input={"pdf": str(pdf_path), "prompts": prompt_versions}
            ) as root:

                result["trace_id"] = current_trace_id()
                self.state["trace_id"] = result["trace_id"]

                paper_text = read_pdf(pdf_path)[:config.MAX_PAPER_CHARS]

                if simulate_failure:
                    paper_text += "\nFAIL_SUMMARY"

                try:

                    # -------------------------
                    # Validation
                    # -------------------------

                    v = await validation.execute(paper_text)

                    self.completed.append(validation)

                    self.update_state(
                        validation.name,
                        v
                    )

                    # -------------------------
                    # Review
                    # -------------------------

                    r = await review.execute(paper_text)

                    self.completed.append(review)

                    self.update_state(
                        review.name,
                        r
                    )

                    # -------------------------
                    # Summary
                    # -------------------------

                    s = await summary.execute(paper_text)

                    self.completed.append(summary)

                    self.update_state(
                        summary.name,
                        s
                    )

                    self.state["status"] = "COMPLETED"

                    save_state(self.state)

                    print("\n================================")
                    print("SAGA SUCCESS")
                    print("================================")

                    print("\nValidation")
                    print(v["validation"])

                    print("\nReview")
                    print(r["review_comments"])

                    print("\nSummary")
                    print(s["summary"])

                    # -------------------------
                    # Evaluation
                    # -------------------------

                    if evaluate:

                        print("\n[Evaluation]")

                        result["scores"] = await evaluate_run(
                            paper_text,
                            self.state["metadata"],
                            use_ragas=use_ragas
                        )

                        write_output(
                            "evaluation.json",
                            json.dumps(result["scores"], indent=4)
                        )

                        for name, s_ in result["scores"].items():
                            print(f"  {name:28s} {s_['value']:.3f}")

                    if root:
                        root.update(output={
                            "status": "COMPLETED",
                            "validation": v["validation"],
                            "review": r["review_comments"],
                            "summary": s["summary"]
                        })

                except Exception as e:

                    error = describe_error(e)

                    print("\nFailure:", error)

                    result["error"] = error

                    self.state["status"] = "FAILED"

                    save_state(self.state)

                    self.rollback()

                    if root:
                        root.update(
                            level="ERROR",
                            status_message=error,
                            output={"status": "ROLLED_BACK", "error": error}
                        )

        # -------------------------
        # Scores -> Langfuse
        # -------------------------

        trace_id = result["trace_id"]

        result["scores"]["saga_success"] = {
            "value": 1.0 if self.state["status"] == "COMPLETED" else 0.0,
            "comment": result["error"]
        }

        # Experiments push scores through their own evaluators instead
        if push_scores:
            for name, s_ in result["scores"].items():
                score(trace_id, name, s_["value"], comment=s_.get("comment"))

        result["status"] = self.state["status"]
        result["metadata"] = self.state["metadata"]
        result["trace_url"] = trace_url(trace_id)

        return result

# -----------------------------------
# Run
# -----------------------------------

async def run_saga(pdf_path, model=None, evaluate=True, use_ragas=True,
                   simulate_failure=False, session_id=None, push_scores=True):

    orchestrator = SagaOrchestrator(model=model, session_id=session_id)

    return await orchestrator.run(
        pdf_path,
        evaluate=evaluate,
        use_ragas=use_ragas,
        simulate_failure=simulate_failure,
        push_scores=push_scores
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Research paper review Saga")
    parser.add_argument("pdf", nargs="?", default="paper.pdf")
    parser.add_argument("--model", help="Override the model for all agents")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--no-ragas", action="store_true", help="Only run custom evaluators")
    parser.add_argument("--simulate-failure", action="store_true",
                        help="Make the Summary agent fail to demo compensation")
    args = parser.parse_args()

    result = asyncio.run(run_saga(
        args.pdf,
        model=args.model,
        evaluate=not args.no_eval,
        use_ragas=not args.no_ragas,
        simulate_failure=args.simulate_failure
    ))

    flush()

    if result["trace_url"]:
        print("\nLangfuse trace:", result["trace_url"])
