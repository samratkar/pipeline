from dataclasses import dataclass
from typing import Any, Optional

import config
from observability import get_langfuse

# -----------------------------------
# Prompt Management (Langfuse)
# -----------------------------------
#
# Agent instructions are versioned in Langfuse. The defaults below are
# used as the fallback (and are what `python seed_prompts.py` uploads
# as version 1).

DEFAULT_PROMPTS = {

    "paper-validation-agent": (
        "You validate research papers. "
        "Briefly decide whether the paper looks academically valid. "
        "Return verdict VALID or INVALID, a short justification and "
        "a confidence between 0 and 1."
    ),

    "paper-review-agent": (
        "Act as an academic reviewer. "
        "Provide concise review comments with:\n"
        "1. Strengths\n"
        "2. Weaknesses\n"
        "3. Suggestions\n"
        "Only make claims that are supported by the paper text."
    ),

    "paper-summary-agent": (
        "Summarize this research paper in approximately 300 words. "
        "Only use information present in the paper text."
    ),
}


@dataclass
class AgentPrompt:

    name: str
    text: str
    version: Optional[int]
    model: Optional[str]
    client: Any = None          # Langfuse prompt client, used to link generations

    @property
    def label(self):

        return f"{self.name}@v{self.version}" if self.version else f"{self.name}@local"


def get_agent_prompt(name):

    default = DEFAULT_PROMPTS[name]

    lf = get_langfuse()

    if lf is None:
        return AgentPrompt(name, default, None, None)

    prompt = lf.get_prompt(
        name,
        label=config.PROMPT_LABEL,
        fallback=default,
        cache_ttl_seconds=60
    )

    if prompt.is_fallback:
        print(f"Prompt '{name}' not found in Langfuse - using local default")
        return AgentPrompt(name, default, None, None)

    return AgentPrompt(
        name=name,
        text=prompt.compile(),
        version=prompt.version,
        model=(prompt.config or {}).get("model"),
        client=prompt
    )
