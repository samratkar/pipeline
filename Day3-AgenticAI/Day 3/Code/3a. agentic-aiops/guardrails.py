import re

from agents import (
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
)

# -----------------------------------
# Guardrails (OpenAI Agents SDK)
# -----------------------------------
#
# A tripped guardrail raises InputGuardrailTripwireTriggered /
# OutputGuardrailTripwireTriggered, which the Saga treats as a failure
# and compensates. Guardrail results are visible in the Langfuse trace.

MIN_PAPER_CHARS = 500

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) instructions",
    r"disregard (all |any )?(the )?(previous|prior|above)",
    r"reveal your (system )?(instructions|prompt)",
    r"mark (this|the) paper as valid",
]

SUMMARY_MIN_WORDS = 150
SUMMARY_MAX_WORDS = 500


def _text(input_data):

    if isinstance(input_data, str):
        return input_data

    return " ".join(str(item.get("content", "")) for item in input_data)


# run_in_parallel=False -> the check runs before the LLM is called,
# so a blocked paper costs no tokens.

@input_guardrail(name="paper_length_check", run_in_parallel=False)
def paper_length_guardrail(ctx, agent, input_data):

    text = _text(input_data)

    too_short = len(text.strip()) < MIN_PAPER_CHARS

    return GuardrailFunctionOutput(
        output_info={"chars": len(text), "min_chars": MIN_PAPER_CHARS},
        tripwire_triggered=too_short
    )


@input_guardrail(name="prompt_injection_check", run_in_parallel=False)
def prompt_injection_guardrail(ctx, agent, input_data):

    text = _text(input_data).lower()

    matches = [p for p in INJECTION_PATTERNS if re.search(p, text)]

    return GuardrailFunctionOutput(
        output_info={"matched_patterns": matches},
        tripwire_triggered=bool(matches)
    )


@output_guardrail(name="review_structure_check")
def review_structure_guardrail(ctx, agent, output):

    missing = [
        section for section in ("strengths", "weaknesses", "suggestions")
        if not getattr(output, section)
    ]

    return GuardrailFunctionOutput(
        output_info={"missing_sections": missing},
        tripwire_triggered=bool(missing)
    )


@output_guardrail(name="summary_length_check")
def summary_length_guardrail(ctx, agent, output):

    words = len(str(output).split())

    return GuardrailFunctionOutput(
        output_info={
            "words": words,
            "allowed": [SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS]
        },
        tripwire_triggered=not (SUMMARY_MIN_WORDS <= words <= SUMMARY_MAX_WORDS)
    )


INPUT_GUARDRAILS = [paper_length_guardrail, prompt_injection_guardrail]
