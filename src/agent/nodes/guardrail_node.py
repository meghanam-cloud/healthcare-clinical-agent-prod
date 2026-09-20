"""
Node: validates the raw user question against security/scope rules before
anything else runs. This is the graph's entry point.
"""

import logging
from src.agent.state import AgentState
from src.agent import guardrails
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)


def guardrail_node(state: AgentState) -> AgentState:
    trace = state.get("trace")
    question = state["question"]

    with trace_span(trace, "guardrail_check", input=question) as span:
        result = guardrails.evaluate(question)
        span.update(
            output={"allowed": result.allowed, "reason": result.reason},
            metadata={"blocked": not result.allowed},
        )

    if not result.allowed:
        logger.info("Guardrail blocked question: %s", result.reason)

    return {
        **state,
        "blocked": not result.allowed,
        "block_reason": result.reason,
    }
