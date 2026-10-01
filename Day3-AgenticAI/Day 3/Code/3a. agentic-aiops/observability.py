from contextlib import contextmanager, nullcontext

import config

# -----------------------------------
# Observability / Tracing (Langfuse)
# -----------------------------------
#
# Every Agents SDK run (agent spans, LLM generations, guardrails,
# handoffs, tokens, latency, errors) is exported to Langfuse through
# the OpenInference instrumentor for the OpenAI Agents SDK.
#
# If Langfuse keys are not set in .env, everything below becomes a
# no-op and the pipeline still runs.

_langfuse = None


def init_observability():

    global _langfuse

    if _langfuse is not None or not config.LANGFUSE_ENABLED:
        return _langfuse

    from langfuse import get_client
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    _langfuse = get_client()

    OpenAIAgentsInstrumentor().instrument()

    if _langfuse.auth_check():
        print("Langfuse tracing enabled")
    else:
        print("WARNING: Langfuse authentication failed - check LANGFUSE_* keys")

    return _langfuse


def get_langfuse():

    return init_observability()


@contextmanager
def observation(name, as_type="span", **kwargs):
    """Langfuse span (or no-op) that nests everything created inside it."""

    lf = get_langfuse()

    if lf is None:
        yield None
        return

    with lf.start_as_current_observation(name=name, as_type=as_type, **kwargs) as span:
        yield span


def trace_attributes(**kwargs):
    """Trace level attributes: session_id, tags, metadata, version, prompt link."""

    if get_langfuse() is None:
        return nullcontext()

    from langfuse import propagate_attributes

    return propagate_attributes(**kwargs)


def current_trace_id():

    lf = get_langfuse()

    return lf.get_current_trace_id() if lf else None


def trace_url(trace_id):

    lf = get_langfuse()

    if lf is None or trace_id is None:
        return None

    return lf.get_trace_url(trace_id=trace_id)


def score(trace_id, name, value, comment=None, data_type=None):

    lf = get_langfuse()

    if lf is None or trace_id is None:
        return

    lf.create_score(
        trace_id=trace_id,
        name=name,
        value=value,
        comment=comment,
        data_type=data_type
    )


def flush():

    lf = get_langfuse()

    if lf is not None:
        lf.flush()
