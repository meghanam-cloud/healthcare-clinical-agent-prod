"""LangGraph orchestration for the clinical agentic RAG workflow.

Flow:

    guardrail
        -> supervisor_plan
        -> execute planned tools (SQL, RAG, or both)
        -> synthesize draft answer
        -> reviewer

The reviewer can:
    pass         -> END
    retry_tools  -> execute selected tools again
    update_plan  -> supervisor creates a revised plan

The reviewer loop is capped at three attempts per user turn.
"""

import time
import logging
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes.guardrail_node import guardrail_node
from src.agent.nodes.supervisor_node import supervisor_node
from src.agent.nodes.tool_executor import tool_executor_node
from src.agent.nodes.response_formatter import response_formatter_node
from src.agent.nodes.review_node import review_node, route_after_review
from src.observability.mlflow_tracker import log_turn
from src.observability.langfuse_tracer import get_tracer
from src.utils.config_loader import settings

logger = logging.getLogger(__name__)

MAX_REVIEW_ATTEMPTS = 3


def _route_after_guardrail(state: AgentState) -> str:
    return "blocked" if state.get("blocked") else "allowed"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("response_formatter", response_formatter_node)
    graph.add_node("review", review_node)
    graph.add_node("blocked_response", response_formatter_node)

    graph.set_entry_point("guardrail")

    graph.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {"blocked": "blocked_response", "allowed": "supervisor"},
    )

    graph.add_edge("supervisor", "tool_executor")
    graph.add_edge("tool_executor", "response_formatter")
    graph.add_edge("response_formatter", "review")

    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "done": END,
            "update_plan": "supervisor",
            "retry_tools": "tool_executor",
        },
    )

    graph.add_edge("blocked_response", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    question: str,
    session_id: str,
    user_id: str,
    chat_history: list,
    trace=None,
) -> dict:
    """Run one complete agentic turn with a maximum of three review attempts."""
    start = time.time()
    graph = get_graph()

    if trace is None:
        trace = get_tracer().trace(
            name="clinical_agent_turn",
            user_id=user_id,
            session_id=session_id,
            input=question,
            metadata={"chat_history_len": len(chat_history or [])},
        )

    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "question": question,
        "chat_history": chat_history,
        "blocked": False,
        "block_reason": "",
        "review_attempts": 0,
        "max_review_attempts": MAX_REVIEW_ATTEMPTS,
        "review_retry_tools": [],
        "plan_revision": 0,
        "trace": trace,
    }

    final_state = graph.invoke(initial_state)

    try:
        trace.update(
            output=final_state.get("final_response", ""),
            metadata={
                "latency_seconds": time.time() - start,
                "blocked": final_state.get("blocked", False),
                "route": final_state.get("route"),
                "plan_revision": final_state.get("plan_revision", 0),
                "review_attempts": final_state.get("review_attempts", 0),
                "review_decision": final_state.get("review_decision"),
                "review_passed": final_state.get("review_passed"),
                "sql_error": final_state.get("sql_error"),
                "rag_error": final_state.get("rag_error"),
            },
        )
    except Exception:
        pass

    plan_text = final_state.get("plan", {})
    log_turn(
        question=question,
        system_prompt=str(plan_text),
        sql_query=final_state.get("sql_query", ""),
        response=final_state.get("final_response", ""),
        model_name=settings.llm.endpoint_name,
        latency_seconds=time.time() - start,
        sql_row_count=len(final_state.get("sql_rows", []) or []),
        guardrail_blocked=final_state.get("blocked", False),
        error=final_state.get("sql_error") or final_state.get("rag_error"),
    )

    return final_state
