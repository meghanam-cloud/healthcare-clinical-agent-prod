"""
Thin wrapper around the Databricks SDK secrets API.

Resolution order for any secret key:
  1. Databricks secret scope (via WorkspaceClient) - used when running inside
     a Databricks App / job where a workspace auth context is available.
  2. OS environment variable of the same name - used for local dev via .env.

Fails soft: if the SDK call errors OR hangs (no workspace context, bad host,
scope/key missing), we fall back to the env var instead of blocking the app.
A hard timeout wraps every SDK call so a misconfigured/unreachable host can
never hang the process - it always falls back within SECRET_LOOKUP_TIMEOUT_SECONDS.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)

SECRET_LOOKUP_TIMEOUT_SECONDS = 5

_workspace_client = None
_client_init_attempted = False
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="secret-lookup")


def _get_client():
    """Lazily create a single WorkspaceClient for the process."""
    global _workspace_client, _client_init_attempted
    if _workspace_client is not None or _client_init_attempted:
        return _workspace_client

    _client_init_attempted = True
    try:
        from databricks.sdk import WorkspaceClient
        _workspace_client = WorkspaceClient()
    except Exception as e:
        logger.warning("WorkspaceClient unavailable, will use env vars only: %s", e)
        _workspace_client = None
    return _workspace_client


def _fetch_from_scope(scope: str, key: str) -> str:
    client = _get_client()
    if client is None:
        raise RuntimeError("No WorkspaceClient available")
    secret_response = client.secrets.get_secret(scope=scope, key=key)
    import base64
    return base64.b64decode(secret_response.value).decode("utf-8")


def get_secret(scope: str, key: str, env_fallback: str) -> str:
    """
    Fetch `key` from the given Databricks secret `scope`, with a hard
    timeout so an unreachable/misconfigured workspace never blocks startup.
    Falls back to os.environ[env_fallback] if the scope/key can't be read
    in time. Raises ValueError if neither source has the value.
    """
    try:
        future = _executor.submit(_fetch_from_scope, scope, key)
        return future.result(timeout=SECRET_LOOKUP_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        logger.warning("Secret scope lookup timed out for %s/%s after %ss, using env fallback", scope, key, SECRET_LOOKUP_TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("Secret scope lookup failed for %s/%s: %s", scope, key, e)

    value = os.environ.get(env_fallback)
    if value is None:
        raise ValueError(
            f"Secret '{key}' not found in scope '{scope}' and env var "
            f"'{env_fallback}' is not set."
        )
    return value
