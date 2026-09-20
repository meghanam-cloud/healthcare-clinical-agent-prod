from src.agent.nodes.supervisor_node import _fallback_plan, _format_history


def test_supervisor_history_includes_previous_sql():
    history = [
        {"role": "user", "content": "Which diabetic patients have HbA1c above 8.5%?"},
        {
            "role": "assistant",
            "content": "23 patients match.",
            "sql_query": "SELECT patient_id FROM healthcare_patient_tbl WHERE hba1c_level > 8.5",
        },
    ]

    formatted = _format_history(history)
    assert "23 patients match." in formatted
    assert "Previous SQL used in this turn" in formatted
    assert "hba1c_level > 8.5" in formatted


def test_follow_up_fallback_uses_prior_sql_context():
    history = [
        {"role": "user", "content": "Which diabetic patients have HbA1c above 8.5%?"},
        {
            "role": "assistant",
            "content": "23 patients match.",
            "sql_query": "SELECT patient_id FROM healthcare_patient_tbl WHERE hba1c_level > 8.5",
        },
    ]

    plan = _fallback_plan("What about their kidney function?", history)
    assert [step["tool"] for step in plan["steps"]] == ["sql"]
