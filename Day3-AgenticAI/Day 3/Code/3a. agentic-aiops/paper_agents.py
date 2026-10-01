from typing import List, Literal

from agents import Agent
from pydantic import BaseModel

import config
from guardrails import (
    INPUT_GUARDRAILS,
    review_structure_guardrail,
    summary_length_guardrail,
)
from prompts import get_agent_prompt

# -----------------------------------
# Agent Framework / Runtime (OpenAI Agents SDK)
# -----------------------------------


class ValidationResult(BaseModel):

    verdict: Literal["VALID", "INVALID"]
    justification: str
    confidence: float


class ReviewResult(BaseModel):

    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]


def build_agents(model_override=None):
    """Create the three agents from the current Langfuse prompt versions.

    Returns {agent_name: (Agent, AgentPrompt)}
    """

    def model_for(prompt):

        return model_override or prompt.model or config.AGENT_MODEL

    validation_prompt = get_agent_prompt("paper-validation-agent")
    review_prompt = get_agent_prompt("paper-review-agent")
    summary_prompt = get_agent_prompt("paper-summary-agent")

    validation_agent = Agent(
        name="ValidationAgent",
        instructions=validation_prompt.text,
        model=model_for(validation_prompt),
        output_type=ValidationResult,
        input_guardrails=INPUT_GUARDRAILS,
    )

    review_agent = Agent(
        name="ReviewAgent",
        instructions=review_prompt.text,
        model=model_for(review_prompt),
        output_type=ReviewResult,
        input_guardrails=INPUT_GUARDRAILS,
        output_guardrails=[review_structure_guardrail],
    )

    summary_agent = Agent(
        name="SummaryAgent",
        instructions=summary_prompt.text,
        model=model_for(summary_prompt),
        input_guardrails=INPUT_GUARDRAILS,
        output_guardrails=[summary_length_guardrail],
    )

    return {
        "ValidationAgent": (validation_agent, validation_prompt),
        "ReviewAgent": (review_agent, review_prompt),
        "SummaryAgent": (summary_agent, summary_prompt),
    }


def format_validation(result: ValidationResult):

    return (
        f"{result.verdict}\n\n"
        f"{result.justification}\n\n"
        f"Confidence: {result.confidence:.2f}"
    )


def format_review(result: ReviewResult):

    def section(title, items):
        return title + "\n" + "\n".join(f"- {item}" for item in items)

    return "\n\n".join([
        section("Strengths", result.strengths),
        section("Weaknesses", result.weaknesses),
        section("Suggestions", result.suggestions),
    ])
