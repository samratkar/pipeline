import os

from dotenv import load_dotenv

# -----------------------------------
# Configuration / Secrets (.env)
# -----------------------------------

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model that powers the Validation, Review and Summary agents.
# A Langfuse prompt can override it through its config: {"model": "..."}
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5-nano")

# Model used by RAGAS / LLM-based evaluators
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")

# Langfuse (observability, prompt management, experiments)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# Which Langfuse prompt label the agents should use (production, staging, ...)
PROMPT_LABEL = os.getenv("PROMPT_LABEL", "production")

# Version tag attached to every trace, used to compare app versions
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

# Characters of the paper sent to each agent
MAX_PAPER_CHARS = int(os.getenv("MAX_PAPER_CHARS", "6000"))

STATE_FILE = "saga_state.json"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
