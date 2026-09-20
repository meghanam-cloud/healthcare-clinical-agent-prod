"""
Long-term memory: durable per-user preferences (e.g. preferred category,
budget range) that persist across sessions. Simple key/value upsert.
"""

import logging
from datetime import datetime, timezone

from src.utils.config_loader import settings
from src.utils.sql_connection import run_query

logger = logging.getLogger(__name__)

TABLE = settings.memory.long_term_full_name


def upsert_preference(user_id: str, key: str, value: str):
    try:
        run_query(
            f"""
            MERGE INTO {TABLE} AS t
            USING (SELECT %s AS user_id, %s AS preference_key, %s AS preference_value) AS s
            ON t.user_id = s.user_id AND t.preference_key = s.preference_key
            WHEN MATCHED THEN UPDATE SET preference_value = s.preference_value, updated_at = %s
            WHEN NOT MATCHED THEN INSERT (user_id, preference_key, preference_value, updated_at)
                VALUES (s.user_id, s.preference_key, s.preference_value, %s)
            """,
            params=(user_id, key, value, datetime.now(timezone.utc), datetime.now(timezone.utc)),
            fetch=False,
        )
    except Exception as e:
        logger.warning("Long-term memory write failed (non-fatal): %s", e)


def get_preferences(user_id: str) -> dict:
    try:
        columns, rows = run_query(
            f"SELECT preference_key, preference_value FROM {TABLE} WHERE user_id = %s",
            params=(user_id,),
        )
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        logger.warning("Long-term memory read failed, continuing with no preferences: %s", e)
        return {}
