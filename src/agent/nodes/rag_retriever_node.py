"""
Node: the RAG worker's retrieval step. Calls the hybrid-search-with-rerank
tool against the clinical guidelines AI Search index and assembles the
retrieved chunks into a grounded context string for generation.
"""

import logging
from src.agent.state import AgentState
from src.agent.tools.rag_tools import hybrid_search_with_rerank
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        source = c.get("source", "unknown")
        page = c.get("page", "unknown")
        parts.append(f"[Context {i} | Source: {source} | Page: {page}]\n{c.get('text', '')}")
    return "\n\n".join(parts)


def rag_retriever_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    question = state["question"]
    plan_query = question
    for step in state.get("plan", {}).get("steps", []):
        if step.get("tool") == "rag" and step.get("query"):
            plan_query = step["query"]
            break

    if state.get("execution_scope") == "retry" and state.get("review_notes"):
        plan_query = (
            f"{plan_query}\nFocus on this reviewer gap when retrieving evidence: "
            f"{state['review_notes']}"
        )

    with trace_span(
        trace,
        "rag_retrieval",
        input={"query": plan_query, "chat_history_len": len(state.get("chat_history", []))},
    ) as span:
        result = hybrid_search_with_rerank.invoke({"query": plan_query})
        chunks = result.get("chunks", [])
        context = _build_context(chunks)
        span.update(
            output=context[:2000],
            metadata={"chunk_count": len(chunks), "error": result.get("error")},
        )

    return {**state, "rag_context": context, "rag_sources": chunks, "rag_error": result.get("error")}
