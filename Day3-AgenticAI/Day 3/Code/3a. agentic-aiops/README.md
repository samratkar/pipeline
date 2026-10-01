# Agentic AIOps – Research Paper Review Saga with AgentOps

Three agents (Validation → Review → Summary) run as a Saga: if any step fails,
completed steps are compensated in reverse order. Run steps are in `steps`.

| AgentOps area | Tool | Where |
|---|---|---|
| Agent framework / runtime | OpenAI Agents SDK | `paper_agents.py` – `Agent`s with structured `output_type`s |
| Agent orchestration | OpenAI Agents SDK + Saga | `orchestrator.py` – `Runner.run` per step, compensation on failure |
| LLM model | OpenAI API | `AGENT_MODEL` in `.env`, or `{"model": ...}` in the Langfuse prompt config |
| Document processing | PyMuPDF / PyPDF | `document.py` – PyMuPDF, falling back to PyPDF |
| Observability / tracing | Langfuse | `observability.py` – OpenInference instrumentor exports agent spans, LLM calls, guardrails, tokens, latency and errors |
| Agent evaluation | RAGAS + custom evaluators | `evaluation.py` – faithfulness, summary score, length, review completeness; stored as Langfuse scores |
| Guardrails | Agents SDK guardrails | `guardrails.py` – paper length and prompt-injection input checks; review-structure and summary-length output checks |
| Experiment tracking | Langfuse datasets / experiments | `experiment.py` – dataset runs compared by model, prompt version and app version |
| Prompt management | Langfuse | `prompts.py`, `seed_prompts.py` – versioned prompts loaded by label and linked to generations |
| Application / demo UI | Streamlit | `app.py` – upload, results, scores, trace link, 👍/👎 feedback score |
| Configuration / secrets | .env | `config.py`, `.env.example` |

## What a trace looks like in Langfuse

```
paper-review-saga (chain)          tags: saga, <model>   session, version, prompt versions
├── extract-pdf                    extractor, pages, chars
├── ValidationAgent → generation   ← linked to prompt paper-validation-agent vN
├── ReviewAgent     → generation + guardrail spans
├── SummaryAgent    → generation + guardrail spans
├── evaluation (evaluator)
└── saga-rollback → compensate-*   (only on failure, trace marked ERROR)
scores: saga_success, validation_*, review_*, summary_*, user_feedback
```

## Things to try in the workshop

- **Rollback:** `python orchestrator.py --simulate-failure`, then open the trace to see the compensation spans.
- **Guardrail:** add "Ignore all previous instructions" to a paper's text. The input guardrail trips before any tokens are spent.
- **Prompt versioning:** edit `paper-summary-agent` in Langfuse, give the new version the `production` label, and rerun. Nothing in the code changes.
- **Experiments:** `python experiment.py`, then `python experiment.py --model gpt-4o-mini`, and compare the two runs in Langfuse.
