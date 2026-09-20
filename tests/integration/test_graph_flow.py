"""Integration tests for the full agentic graph.

Requires a real Databricks SQL Warehouse, model serving endpoint and RAG index.
"""

import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run against real Databricks resources.",
)


def test_blocked_question_short_circuits():
    from src.agent.graph import run_agent

    result = run_agent(
        question="what's the weather today?",
        session_id="test-session",
        user_id="test-user",
        chat_history=[],
    )
    assert result["blocked"] is True
    assert result["final_response"]


def test_sql_question_creates_sql_plan_and_answer():
    from src.agent.graph import run_agent

    result = run_agent(
        question="How many patients have Type 2 Diabetes?",
        session_id="test-session",
        user_id="test-user",
        chat_history=[],
    )

    assert result["blocked"] is False
    assert [s["tool"] for s in result["plan"]["steps"]] == ["sql"]
    assert "select" in result["sql_query"].lower()
    assert result["final_response"]
    assert result["review_passed"] is True


def test_rag_question_creates_rag_plan_and_answer():
    from src.agent.graph import run_agent

    result = run_agent(
        question="What does the clinical guideline say about HbA1c above 8.5%?",
        session_id="test-session",
        user_id="test-user",
        chat_history=[],
    )

    assert result["blocked"] is False
    assert [s["tool"] for s in result["plan"]["steps"]] == ["rag"]
    assert result["rag_sources"]
    assert result["final_response"]
    assert result["review_passed"] is True


def test_combined_question_uses_sql_and_rag():
    from src.agent.graph import run_agent

    result = run_agent(
        question=(
            "Which patients are taking both Lisinopril and Losartan, "
            "and what does the clinical guideline say about this combination?"
        ),
        session_id="test-session",
        user_id="test-user",
        chat_history=[],
    )

    assert result["blocked"] is False
    assert {s["tool"] for s in result["plan"]["steps"]} == {"sql", "rag"}
    assert result["sql_query"]
    assert result["sql_rows"]
    assert result["rag_sources"]
    assert result["final_response"]
    assert result["review_passed"] is True
    assert result["review_attempts"] <= 3
