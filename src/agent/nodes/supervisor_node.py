"""Supervisor node: create an explicit execution plan before any worker runs."""

import json
import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import AgentState
from src.utils.llm_client import get_llm
from src.utils.config_loader import settings
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the supervisor of a healthcare agentic RAG system.
Your job is to PLAN how the user's question should be answered. Do not answer the
question yourself and do not generate SQL.

Available tools:
1. sql - queries the de-identified patient cohort table. Use it for patient-level
   data, filtering, counts, cohorts, averages, risk tiers, medications on file,
   labs, adherence, ED visits, hospitalizations, or other structured patient data.
2. rag - searches the clinical guidelines manual. Use it for guideline/protocol
   thresholds, clinical care actions, medication safety rules, contraindications,
   treatment pathways, escalation rules, and other information in the manual.

Use BOTH tools when the question requires patient data AND interpretation against
clinical guidelines. Do not use a tool merely because it is available.

The clinical manual explicitly contains risk stratification, diabetes, hypertension,
CKD/respiratory protocols, medication safety, adherence, and care-escalation rules.
The structured table contains patient demographics, conditions, biomarkers, blood
pressure, medications, allergies, adherence, utilization, risk tier and readmission
probability.

Return ONLY valid JSON with this shape:
{
  "objective": "what must be answered",
  "steps": [
    {
      "tool": "sql" | "rag",
      "purpose": "why this tool is needed",
      "query": "the focused query/instruction for that tool"
    }
  ],
  "answer_requirements": ["specific facts the final answer must contain"]
}

Rules:
- Include one or two steps only; each tool can appear at most once.
- If the question is purely structured patient data, choose sql only.
- If it is purely guideline/protocol knowledge, choose rag only.
- If it asks something like "which patients ... and what does the guideline say",
  choose both.
- The SQL query field is NOT SQL. It is a plain-language instruction describing
  what data must be retrieved.
- The RAG query should be a focused retrieval query using the clinical concepts
  needed to answer the question.
- Do not invent a table name, column name, patient value, or guideline fact.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def _fallback_plan(question: str, chat_history: list[dict] | None = None) -> dict:
    """Safe deterministic fallback, including follow-up context."""
    lowered = question.lower()
    chat_history = chat_history or []
    history_text = " ".join(str(t.get("content", "")) for t in chat_history[-6:]).lower()
    prior_sql = any(bool(t.get("sql_query")) for t in chat_history[-6:])

    # Short follow-ups such as "what about their kidney function?" are usually
    # anchored to the prior patient/cohort result. Prefer SQL when persisted
    # history contains a previous SQL-backed turn.
    follow_up_terms = (
        "those patients", "these patients", "their ", "them", "that cohort",
        "those results", "these results", "what about", "and what about",
    )
    is_follow_up = any(t in lowered for t in follow_up_terms) and bool(chat_history)
    sql_terms = (
        "patient", "patients", "cohort", "count", "how many", "which patients",
        "list", "average", "adherence", "ed visit", "hospitalization", "risk tier",
        "readmission", "lab", "hba1c", "egfr", "medications on file",
    )
    rag_terms = (
        "guideline", "protocol", "recommend", "should", "threshold", "contraindication",
        "dosage", "dosing", "treatment", "manage", "escalation", "first-line",
        "safety", "medication", "ckd", "diabetes", "hypertension", "asthma", "copd",
    )
    sql_hit = any(t in lowered for t in sql_terms)
    rag_hit = any(t in lowered for t in rag_terms)

    if is_follow_up and prior_sql:
        sql_hit = True

    # Preserve a prior guideline context for vague follow-ups such as
    # "what about the guideline for them?" while still allowing SQL + RAG.
    if is_follow_up and any(t in history_text for t in ("guideline", "protocol", "contraindication", "threshold")):
        rag_hit = rag_hit or any(t in lowered for t in ("guideline", "protocol", "rule", "safe", "allowed"))

    if sql_hit and rag_hit:
        tools = [
            {"tool": "sql", "purpose": "Retrieve the relevant patient/cohort data.", "query": question},
            {"tool": "rag", "purpose": "Retrieve the guideline/protocol rules needed to interpret the patient data.", "query": question},
        ]
    elif sql_hit:
        tools = [{"tool": "sql", "purpose": "Retrieve the requested structured patient data.", "query": question}]
    else:
        tools = [{"tool": "rag", "purpose": "Retrieve the relevant clinical guideline/protocol evidence.", "query": question}]

    return {
        "objective": question,
        "steps": tools,
        "answer_requirements": ["Directly answer the user's question", "Use only retrieved evidence"],
    }


def _validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps or len(steps) > 2:
        raise ValueError("Plan must contain one or two tool steps")

    cleaned = []
    seen = set()
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Each plan step must be an object")
        tool = str(step.get("tool", "")).lower().strip()
        if tool not in {"sql", "rag"} or tool in seen:
            raise ValueError("Plan tools must be unique and limited to sql/rag")
        query = str(step.get("query", "")).strip()
        purpose = str(step.get("purpose", "")).strip()
        if not query:
            raise ValueError(f"Missing query for {tool} step")
        cleaned.append({"tool": tool, "purpose": purpose, "query": query})
        seen.add(tool)

    objective = str(plan.get("objective", "")).strip()
    if not objective:
        objective = "Answer the user's question using the selected evidence."

    requirements = plan.get("answer_requirements", [])
    if not isinstance(requirements, list):
        requirements = [str(requirements)]

    return {
        "objective": objective,
        "steps": cleaned,
        "answer_requirements": [str(x) for x in requirements if str(x).strip()],
    }


def _format_history(chat_history: list[dict]) -> str:
    """Format recent persisted conversation for planning.

    The history may contain the SQL used in previous assistant turns. Keeping
    that query is important for follow-ups such as "what about those patients?"
    because the natural-language answer may summarize away the exact cohort
    definition used in the previous turn.
    """
    if not chat_history:
        return "No previous conversation is available."

    parts = []
    for i, turn in enumerate(chat_history[-8:], start=1):
        role = str(turn.get("role", "unknown")).capitalize()
        content = str(turn.get("content", "")).strip()
        parts.append(f"Turn {i} | {role}:\n{content}")
        sql_query = turn.get("sql_query")
        if role.lower() == "assistant" and sql_query:
            parts.append(f"Previous SQL used in this turn:\n{sql_query}")
    return "\n\n".join(parts)


def supervisor_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    question = state["question"]
    revision = state.get("plan_revision", 0) + 1
    is_revision = bool(state.get("review_notes")) and bool(state.get("plan"))
    history_context = _format_history(state.get("chat_history", []))

    previous_context = ""
    if is_revision:
        previous_context = (
            "\nPrevious plan:\n"
            f"{json.dumps(state.get('plan', {}), indent=2)}\n"
            "\nReviewer feedback:\n"
            f"{state.get('review_notes', '')}\n"
            "\nPrevious draft:\n"
            f"{state.get('draft_response', '')}\n"
            "\nCreate a revised plan that fixes the reviewer's identified gap.\n"
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Recent conversation from short-term memory:\n{history_context}\n\n"
                f"Current user question:\n{question}\n"
                f"{previous_context}"
            )
        ),
    ]

    with trace_span(
        trace,
        "supervisor_plan",
        input={
            "question": question,
            "revision": revision,
            "is_revision": is_revision,
            "chat_history_len": len(state.get("chat_history", [])),
        },
    ) as span:
        gen = span.generation(
            name="llm_supervisor_plan",
            input=[m.content for m in messages],
            model=settings.llm.endpoint_name,
            metadata={
                "revision": revision,
                "is_revision": is_revision,
                "chat_history_len": len(state.get("chat_history", [])),
            },
        )
        try:
            response = get_llm().invoke(messages)
            plan = _validate_plan(_extract_json(response.content))
        except Exception as exc:
            logger.warning("Supervisor planning failed; using deterministic fallback: %s", exc)
            plan = _fallback_plan(question, state.get("chat_history", []))
        gen.update(output=json.dumps(plan))
        gen.end()
        span.update(output=plan, metadata={"revision": revision})

    tools = [s["tool"] for s in plan["steps"]]
    route = "both" if len(tools) == 2 else tools[0]

    return {
        **state,
        "plan": plan,
        "plan_revision": revision,
        "route": route,
        "execution_scope": "plan",
        "retry_tools": [],
    }

def route_after_supervisor(state: AgentState) -> str:
    return "execute"
