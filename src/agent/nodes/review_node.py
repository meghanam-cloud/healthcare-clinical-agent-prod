"""Reviewer agent: evaluate answer quality and control the next agentic step."""

import json
import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import AgentState
from src.utils.llm_client import get_llm
from src.utils.config_loader import settings
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the final reviewer for a healthcare agentic RAG system.
You receive:
- the user's question
- the supervisor's current plan
- the evidence returned by SQL and/or RAG
- the draft answer
- the current review attempt number

Your job is NOT to invent a better answer from your own knowledge. Decide whether
the current evidence and draft are sufficient to answer the user's question.

Review criteria:
1. COMPLETENESS: Does the answer address every material part of the question?
2. EVIDENCE: Is every factual claim supported by the SQL result or retrieved
   guideline context? Do not use your own medical knowledge as evidence.
3. PLAN FIT: Did the supervisor select the right evidence source(s)? Use SQL for
   patient/cohort facts, RAG for guideline/protocol facts, and BOTH when both are
   required.
4. CLINICAL GROUNDING: Guideline claims must come from the supplied manual
   context. Patient claims must come from the supplied SQL result.
5. SAFETY: Do not allow the draft to invent a diagnosis, personalized treatment,
   prescription, or unsupported clinical action.
6. CITATIONS: If RAG evidence is used, the draft should identify the relevant
   manual page number(s) where available.

Choose exactly one decision:
- "pass": The evidence is sufficient and the draft can be shown to the user.
- "update_plan": The current plan is wrong or incomplete; the supervisor should
  create a revised plan, usually because the question needs a different tool or
  a materially different decomposition.
- "retry_tools": The plan is appropriate, but one or more selected tools should
  be executed again because the query/retrieval was inadequate or failed.

For retry_tools, set retry_tools to one or both of: ["sql", "rag"].
For update_plan, retry_tools should be [].
For pass, retry_tools should be [].

Return ONLY valid JSON:
{
  "decision": "pass" | "update_plan" | "retry_tools",
  "reason": "short explanation",
  "missing_evidence": ["specific missing fact or requirement"],
  "retry_tools": ["sql", "rag"],
  "final_answer": "the grounded final answer when decision=pass; otherwise empty"
}
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


def _sql_evidence(state: AgentState) -> str:
    if state.get("sql_error"):
        return f"SQL ERROR: {state['sql_error']}"
    columns = state.get("sql_columns", [])
    rows = state.get("sql_rows", [])
    if not rows:
        return "No SQL rows returned."
    return "\n".join(
        ", ".join(f"{c}={v}" for c, v in zip(columns, row))
        for row in rows[:50]
    )


def _validate_review(review: dict) -> dict:
    decision = str(review.get("decision", "")).lower().strip()
    if decision not in {"pass", "update_plan", "retry_tools"}:
        raise ValueError("Invalid review decision")

    retry_tools = [str(x).lower() for x in review.get("retry_tools", [])]
    retry_tools = [x for x in retry_tools if x in {"sql", "rag"}]
    if decision == "retry_tools" and not retry_tools:
        raise ValueError("retry_tools decision requires at least one tool")
    if decision != "retry_tools":
        retry_tools = []

    final_answer = str(review.get("final_answer", "")).strip()
    if decision == "pass" and not final_answer:
        raise ValueError("pass decision requires final_answer")

    missing = review.get("missing_evidence", [])
    if not isinstance(missing, list):
        missing = [str(missing)]

    return {
        "decision": decision,
        "reason": str(review.get("reason", "")).strip(),
        "missing_evidence": [str(x) for x in missing],
        "retry_tools": retry_tools,
        "final_answer": final_answer,
    }


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


def review_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    attempt = state.get("review_attempts", 0) + 1
    plan = state.get("plan", {})
    draft = state.get("draft_response", "")

    if not draft.strip():
        # An empty draft is never a pass. Ask the tool loop to re-run the plan.
        decision = "retry_tools" if attempt < state.get("max_review_attempts", 3) else "max_attempts"
        retry_tools = [step.get("tool") for step in plan.get("steps", [])] if decision == "retry_tools" else []
        final = "" if decision == "retry_tools" else (
            "I couldn't produce a sufficiently verified answer after the maximum number of attempts. "
            "Please rephrase the question and try again."
        )
        return {
            **state,
            "review_attempts": attempt,
            "review_passed": False,
            "review_decision": decision,
            "review_notes": "The answer-generation step produced no usable draft.",
            "review_retry_tools": retry_tools,
            "final_response": final,
        }

    tools = [step.get("tool") for step in plan.get("steps", [])]
    evidence = []
    if "sql" in tools:
        evidence.append(
            "=== SQL ===\n"
            f"Query:\n{state.get('sql_query', '')}\n"
            f"Evidence:\n{_sql_evidence(state)}"
        )
    if "rag" in tools:
        evidence.append(
            "=== RAG ===\n"
            + (state.get("rag_context", "") or "No RAG context returned.")
            + (f"\nRAG ERROR: {state.get('rag_error')}" if state.get("rag_error") else "")
        )

    _sep = "\n\n"
    user_content = (
        f"Recent conversation:\n{_format_history(state.get('chat_history', []))}\n\n"
        f"Current question:\n{state['question']}\n\n"
        f"Plan:\n{json.dumps(plan, indent=2)}\n\n"
        f"Evidence:\n{_sep.join(evidence)[:9000]}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Review attempt: {attempt} of {state.get('max_review_attempts', 3)}"
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]

    with trace_span(
        trace,
        "review",
        input={"question": state["question"], "plan": plan, "attempt": attempt},
    ) as span:
        gen = span.generation(
            name="llm_review",
            input=[m.content for m in messages],
            model=settings.llm.endpoint_name,
            metadata={"review_attempt": attempt, "temperature": 0.0},
        )
        try:
            response = get_llm().invoke(messages)
            review = _validate_review(_extract_json(response.content))
        except Exception as exc:
            logger.warning("Review agent failed: %s", exc)
            review = {
                "decision": "retry_tools" if attempt < state.get("max_review_attempts", 3) else "max_attempts",
                "reason": f"Reviewer invocation/parse failed: {exc}",
                "missing_evidence": [],
                "retry_tools": tools if attempt < state.get("max_review_attempts", 3) else [],
                "final_answer": "",
            }
        gen.update(output=json.dumps(review))
        gen.end()
        span.update(output=review, metadata={"review_attempt": attempt})

    decision = review["decision"]
    if attempt >= state.get("max_review_attempts", 3) and decision != "pass":
        decision = "max_attempts"
        final_response = (
            "I couldn't produce a sufficiently verified answer after 3 review attempts. "
            "Please rephrase the question or narrow the requested information."
        )
        passed = False
    else:
        final_response = review.get("final_answer", "") if decision == "pass" else ""
        passed = decision == "pass"

    notes = review.get("reason", "")
    if review.get("missing_evidence"):
        notes += " Missing evidence: " + "; ".join(review["missing_evidence"])

    return {
        **state,
        "review_attempts": attempt,
        "review_passed": passed,
        "review_decision": decision,
        "review_notes": notes,
        "review_retry_tools": review.get("retry_tools", []),
        "final_response": final_response,
    }


def route_after_review(state: AgentState) -> str:
    decision = state.get("review_decision", "max_attempts")
    if decision == "pass":
        return "done"
    if decision == "update_plan":
        return "update_plan"
    if decision == "retry_tools":
        return "retry_tools"
    return "done"
