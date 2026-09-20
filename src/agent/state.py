"""LangGraph state shared by all nodes in the clinical agent."""

from typing import TypedDict, Optional, List, Dict, Any


class ChatTurn(TypedDict, total=False):
    role: str
    content: str
    sql_query: Optional[str]


class AgentState(TypedDict, total=False):
    # input
    session_id: str
    user_id: str
    question: str
    chat_history: List[ChatTurn]

    # guardrails
    blocked: bool
    block_reason: str

    # supervisor plan
    plan: Dict[str, Any]
    plan_revision: int
    route: str  # "sql" | "rag" | "both"

    # execution control
    execution_scope: str  # "plan" | "retry"
    retry_tools: List[str]
    review_attempts: int
    max_review_attempts: int

    # SQL evidence
    schema_context: str
    sql_query: str
    sql_columns: List[str]
    sql_rows: List[Any]
    sql_error: Optional[str]

    # RAG evidence
    rag_context: str
    rag_sources: List[Dict[str, Any]]
    rag_error: Optional[str]

    # answer + review
    draft_response: str
    review_passed: bool
    review_decision: str  # "pass" | "update_plan" | "retry_tools" | "max_attempts"
    review_notes: str
    review_retry_tools: List[str]
    final_response: str

    # observability
    trace: Any
    trace_id: Optional[str]
