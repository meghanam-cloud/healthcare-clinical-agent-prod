"""
Node: execute the SQL produced by the SQL worker and record the result on state.
Reviewer-driven retries are controlled by the outer agentic graph, not here.
"""

import logging
from src.agent.state import AgentState
from src.agent.tools.sql_tools import execute_sql_query
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)


def sql_executor_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    sql_query = state["sql_query"]

    with trace_span(
        trace,
        "sql_execution",
        input=sql_query,
        metadata={"review_attempt": state.get("review_attempts", 0)},
    ) as span:
        result = execute_sql_query.invoke({"query": sql_query})

        if result.get("error"):
            logger.warning("SQL execution error: %s", result["error"])
            span.update(
                output={"error": result["error"]},
                level="ERROR",
                metadata={"review_attempt": state.get("review_attempts", 0)},
            )
            return {**state, "sql_error": result["error"], "sql_columns": [], "sql_rows": []}

        span.update(
            output={
                "columns": result["columns"],
                "row_count": result.get("row_count", len(result["rows"])),
            },
            metadata={
                "review_attempt": state.get("review_attempts", 0),
                "row_count": len(result["rows"]),
                "columns": result["columns"],
            },
        )

    return {
        **state,
        "sql_columns": result["columns"],
        "sql_rows": result["rows"],
        "sql_error": None,
    }
