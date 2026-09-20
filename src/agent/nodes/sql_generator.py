"""Generate one safe Databricks SQL query for the SQL step in the supervisor plan."""

import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import AgentState
from src.utils.llm_client import get_llm
from src.utils.config_loader import settings
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """You are the SQL worker for a healthcare care-coordination agent.
Generate ONE read-only Databricks SQL SELECT statement that satisfies the SQL
step in the supervisor's plan.

The only allowed table is:
{table_name}

Schema and semantic notes:
{schema_context}

Important clinical/data semantics:
- `risk_tier` is the pre-computed risk tier stored in the patient table. Use it
  when the user asks for the stored cohort risk tier.
- If the question asks for patients who meet a guideline-defined high-risk rule,
  evaluate the underlying fields instead of assuming `risk_tier` is identical.
  The clinical manual's Section 8.1 uses HbA1c > 8.5 OR BP >= 150/95 OR eGFR < 45
  as a biometric high-risk flag.
- `blood_pressure` is a STRING such as '160/109'. To compare it numerically,
  split systolic and diastolic components and cast them to INT.
- `med_adherence_rate` is 0-1, so 0.60 means 60%.
- `known_allergies` and `current_medications` are free-text strings. Use
  case-insensitive matching when filtering medication/allergy text.
- For medication safety questions, retrieve the relevant medication/allergy,
  renal, and utilization columns rather than returning every column.
- Do not infer a diagnosis or treatment recommendation from the table. Retrieve
  the requested facts; guideline interpretation is the RAG worker's job.

SQL rules:
1. Output ONLY SQL. No markdown and no explanation.
2. Use SELECT only, one statement, and reference the allowed table.
3. Never INSERT, UPDATE, DELETE, MERGE, DROP, ALTER, TRUNCATE, CREATE, GRANT,
   or access another table.
4. Prefer explicit, human-readable columns. Avoid SELECT * unless the question
   genuinely requires the full row.
5. Use exact configured column names from the schema.
6. For patient lists, include LIMIT 50 unless the user explicitly requests a
   different limit. Counts/aggregates do not need LIMIT.
7. Use NULL-safe logic where appropriate and do not invent column values.
8. If the SQL step is only one part of a combined SQL+RAG plan, retrieve the
   structured facts needed for the final combined answer and leave guideline
   interpretation to RAG.
"""


def _extract_sql(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return text.strip().rstrip(";").strip()


def _sql_step(state: AgentState) -> dict:
    for step in state.get("plan", {}).get("steps", []):
        if step.get("tool") == "sql":
            return step
    return {}


def sql_generator_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    step = _sql_step(state)
    attempt = state.get("review_attempts", 0) + 1
    is_retry = state.get("execution_scope") == "retry" and bool(state.get("sql_query"))

    system_prompt = SYSTEM_TEMPLATE.format(
        table_name=settings.data.full_table_name,
        schema_context=state.get("schema_context", ""),
    )

    messages = [SystemMessage(content=system_prompt)]
    for turn in state.get("chat_history", [])[-6:]:
        role_prefix = "User" if turn["role"] == "user" else "Assistant"
        history_line = f"{role_prefix}: {turn['content']}"
        if turn.get("sql_query"):
            history_line += f"\nPrevious SQL used in this turn: {turn['sql_query']}"
        messages.append(HumanMessage(content=history_line))

    user_prompt = (
        f"Original question: {state['question']}\n\n"
        f"Supervisor SQL step: {step.get('query', state['question'])}"
    )
    if is_retry:
        user_prompt += (
            f"\n\nReviewer feedback requiring a SQL retry:\n{state.get('review_notes', '')}"
            f"\n\nPrevious SQL:\n{state.get('sql_query', '')}"
            f"\nPrevious SQL error, if any:\n{state.get('sql_error', '')}"
            "\nGenerate a corrected query."
        )
    messages.append(HumanMessage(content=user_prompt))

    with trace_span(
        trace,
        "sql_generation",
        input={"question": state["question"], "step": step, "is_retry": is_retry},
        metadata={"review_attempt": attempt, "is_retry": is_retry},
    ) as span:
        gen = span.generation(
            name="llm_sql_generation",
            input=[m.content for m in messages],
            model=settings.llm.endpoint_name,
            metadata={"temperature": settings.llm.temperature, "is_retry": is_retry},
        )
        response = get_llm().invoke(messages)
        sql_query = _extract_sql(response.content)
        gen.update(output=response.content)
        gen.end()
        span.update(output=sql_query, metadata={"sql_query": sql_query})

    logger.info("Generated SQL: %s", sql_query)
    return {
        **state,
        "sql_query": sql_query,
        "sql_error": None,
    }
