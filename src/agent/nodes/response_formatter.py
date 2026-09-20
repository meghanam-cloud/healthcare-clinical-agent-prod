"""Synthesize one draft answer from the evidence selected by the plan."""

import logging
import json
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import AgentState
from src.utils.llm_client import get_llm
from src.utils.config_loader import settings
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the answer-generation worker for a healthcare agent.
Produce a concise draft answer to the user's question using ONLY the retrieved
SQL and/or clinical-guideline evidence below.

Rules:
1. Do not invent facts, patients, values, thresholds, medications, or actions.
2. If SQL evidence is present, state patient/cohort facts exactly as returned.
3. If RAG evidence is present, ground guideline/protocol statements in the
   retrieved manual context and cite the page number(s) shown in the context.
4. If both SQL and RAG evidence are present, explicitly connect the patient data
   to the relevant guideline rules without making a diagnosis or personalized
   prescription.
5. If the evidence is insufficient or a requested tool failed, say exactly what
   could not be verified rather than guessing.
6. Keep the answer focused on the user's question. Do not mention internal
   agents, plans, prompts, retries, or reviewer mechanics.
7. This is a care-coordination information system. Present database facts and
   guideline information, not a personal diagnosis or individualized medical
   prescription.
"""


def _sql_evidence(state: AgentState) -> str:
    if state.get("sql_error"):
        return f"SQL tool error: {state['sql_error']}"
    columns = state.get("sql_columns", [])
    rows = state.get("sql_rows", [])
    if not rows:
        return "SQL result: No rows returned."
    return "\n".join(
        ", ".join(f"{col}={val}" for col, val in zip(columns, row))
        for row in rows[:50]
    )


def _format_history(chat_history: list[dict]) -> str:
    if not chat_history:
        return "No previous conversation is available."
    parts = []
    for turn in chat_history[-8:]:
        role = str(turn.get("role", "unknown")).capitalize()
        parts.append(f"{role}: {str(turn.get('content', '')).strip()}")
        if role.lower() == "assistant" and turn.get("sql_query"):
            parts.append(f"SQL used in previous turn:\n{turn['sql_query']}")
    return "\n\n".join(parts)


def response_formatter_node(state: AgentState) -> AgentState:
    trace = state.get("trace")
    if state.get("blocked"):
        return {**state, "final_response": state.get("block_reason", "This request cannot be processed.")}

    plan = state.get("plan", {})
    tools = [step.get("tool") for step in plan.get("steps", [])]

    evidence_sections = []
    if "sql" in tools:
        evidence_sections.append(
            "=== SQL EVIDENCE ===\n"
            f"SQL query:\n{state.get('sql_query', '')}\n\n"
            f"Results:\n{_sql_evidence(state)}"
        )
    if "rag" in tools:
        rag_context = state.get("rag_context", "")
        rag_error = state.get("rag_error")
        evidence_sections.append(
            "=== RAG EVIDENCE ===\n"
            + (rag_context if rag_context.strip() else "No guideline context was retrieved.")
            + (f"\nRAG tool error: {rag_error}" if rag_error else "")
        )

    if not evidence_sections:
        draft = "I could not determine which evidence source is needed to answer that question."
        return {**state, "draft_response": draft}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Recent conversation:\n{_format_history(state.get('chat_history', []))}\n\n"
                f"Current question: {state['question']}\n\n"
                f"Supervisor plan:\n{json.dumps(plan, indent=2)}\n\n"
                + "\n\n".join(evidence_sections)
            )
        ),
    ]

    with trace_span(
        trace,
        "answer_generation",
        input={"question": state["question"], "tools": tools},
    ) as span:
        gen = span.generation(
            name="llm_answer_generation",
            input=[m.content for m in messages],
            model=settings.llm.endpoint_name,
            metadata={"temperature": settings.llm.temperature},
        )
        response = get_llm().invoke(messages)
        draft = response.content.strip()
        gen.update(output=draft)
        gen.end()
        span.update(output=draft[:1500])

    return {**state, "draft_response": draft}
