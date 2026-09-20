"""
Single shared helper for opening a databricks-sql-connector connection to
the configured SQL Warehouse. Used by both the agent's SQL execution tool
and the memory read/write modules. Databricks Apps have no Spark runtime,
so every query in this codebase goes through this connector - never PySpark.
"""

import logging
from contextlib import contextmanager
from databricks import sql as dbsql

from src.utils.config_loader import settings

logger = logging.getLogger(__name__)


@contextmanager
def get_connection():
    conn = dbsql.connect(
        server_hostname=settings.databricks_host.replace("https://", "").replace("http://", ""),
        http_path=f"/sql/1.0/warehouses/{settings.warehouse_id}",
        access_token=settings.secrets.databricks_token,
    )
    try:
        yield conn
    finally:
        conn.close()


def run_query(query: str, params: tuple = None, fetch: bool = True):
    """Execute `query` and return (columns, rows) if fetch else (None, None)."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params or [])
            if not fetch:
                return None, None
            columns = [c[0] for c in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return columns, rows
