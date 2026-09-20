"""
Short-term (per-session) memory: recent conversation turns, used to give
the LLM context for follow-up questions ("what about in blue?").
"""

import logging
from datetime import datetime, timezone

from src.utils.config_loader import settings
from src.utils.sql_connection import run_query

logger = logging.getLogger(__name__)

TABLE = settings.memory.short_term_full_name


def log_turn(session_id: str, role: str, content: str, sql_query: str = None):
    """Append one turn. Never raises - memory logging must not break the chat."""
    try:
        run_query(
            f"""
            INSERT INTO {TABLE} (session_id, turn_id, role, content, sql_query, created_at)
            VALUES (%s,
                    COALESCE((SELECT MAX(turn_id) FROM {TABLE} WHERE session_id = %s), 0) + 1,
                    %s, %s, %s, %s)
            """,
            params=(session_id, session_id, role, content, sql_query, datetime.now(timezone.utc)),
            fetch=False,
        )
    except Exception as e:
        logger.warning("Short-term memory write failed (non-fatal): %s", e)


def get_recent_turns(session_id: str, limit: int = None) -> list[dict]:
    """Return the last `limit` turns for a session, oldest first."""
    limit = limit or settings.memory.short_term_turns_limit
    try:
        columns, rows = run_query(
            f"""
            SELECT role, content, sql_query, created_at
            FROM {TABLE}
            WHERE session_id = %s
            ORDER BY turn_id DESC
            LIMIT %s
            """,
            params=(session_id, limit),
        )
        turns = [dict(zip(columns, row)) for row in rows]
        return list(reversed(turns))
    except Exception as e:
        logger.warning("Short-term memory read failed, continuing with no history: %s", e)
        return []
