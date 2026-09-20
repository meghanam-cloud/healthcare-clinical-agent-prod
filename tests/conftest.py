"""
Unit tests exercise pure logic (guardrails, SQL safety checks, config
shape) and must not require a real Databricks workspace/secret scope.
Setting these env vars before any `src.*` module is imported lets
config_loader's secret resolution fall through to env vars instantly,
instead of every test file needing to know about that plumbing.
"""

import os

os.environ.setdefault("DATABRICKS_HOST", "https://unit-test.cloud.databricks.com")
os.environ.setdefault("DATABRICKS_WAREHOUSE_ID", "unit-test-warehouse")
os.environ.setdefault("DATABRICKS_TOKEN", "unit-test-token")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "unit-test-public-key")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "unit-test-secret-key")
os.environ.setdefault("LANGFUSE_HOST", "https://unit-test.langfuse.local")
