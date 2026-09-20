"""
Read-only SQL tool for the healthcare patient cohort.

This is the only place in the agent that talks to the SQL Warehouse
for patient-cohort queries.

The tool is wrapped as a LangChain @tool so it can also be bound
directly to an LLM if a future node wants a full tool-calling
ReAct loop instead of the fixed generate -> execute pipeline.
"""

import logging

from langchain_core.tools import tool

from src.utils.config_loader import settings
from src.utils.sql_connection import run_query

logger = logging.getLogger(__name__)

MAX_ROWS = 50


def _is_safe_select(query: str) -> tuple[bool, str]:
    """
    Defense-in-depth validation for generated SQL.

    Only a single read-only SELECT statement is allowed and it must
    reference the configured healthcare patient table.
    """

    stripped = query.strip().rstrip(";")
    lowered = stripped.lower()

    if not stripped:
        return False, "Query cannot be empty."

    if ";" in stripped:
        return False, "Only a single statement is allowed."

    if not lowered.startswith("select"):
        return False, "Only SELECT statements are allowed."

    forbidden = [
        "insert",
        "update",
        "delete",
        "drop",
        "truncate",
        "alter",
        "merge",
        "create",
        "grant",
    ]

    if any(f" {keyword} " in f" {lowered} " for keyword in forbidden):
        return False, "Query contains a disallowed write/DDL keyword."

    configured_table = settings.data.table.lower()

    if configured_table not in lowered:
        return (
            False,
            f"Query must reference the {settings.data.table} table.",
        )

    return True, ""


@tool
def execute_sql_query(query: str) -> dict:
    """
    Execute a read-only SELECT query against the configured
    healthcare patient Delta table through the SQL Warehouse.

    Args:
        query: A single SELECT statement targeting the configured
            healthcare patient table.

    Returns:
        Dictionary containing:
            columns: list of column names
            rows: result rows
            row_count: number of returned rows
            error: error message or None
    """

    is_safe, reason = _is_safe_select(query)

    if not is_safe:
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": reason,
        }

    guarded_query = query.strip().rstrip(";")

    if " limit " not in guarded_query.lower():
        guarded_query = f"{guarded_query} LIMIT {MAX_ROWS}"

    try:
        columns, rows = run_query(guarded_query)

        rows = [list(row) for row in rows][:MAX_ROWS]

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "error": None,
        }

    except Exception as exc:
        logger.warning(
            "SQL execution failed: %s",
            exc,
        )

        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": str(exc),
        }