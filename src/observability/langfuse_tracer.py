"""
Langfuse tracing helpers.

This module provides a small compatibility layer around the Langfuse
Python SDK so the rest of the application does not need to know about
Langfuse client lifecycle details.

The application uses the Langfuse v2 API:

    Langfuse(...)
        -> trace(...)
            -> span(...)
            -> generation(...)

The tracer is intentionally kept in one place so that configuration,
authentication, flushing and error handling can be managed centrally.

Usage:

    from src.observability.langfuse_tracer import get_tracer, trace_span

    tracer = get_tracer()

    trace = tracer.trace(
        name="clinical_agent_turn",
        user_id=user_id,
        session_id=session_id,
        input=question,
    )

    with trace_span(trace, "guardrail_check", input=question) as span:
        ...
        span.update(output=result)

    tracer.flush()
"""

import logging
from contextlib import contextmanager

from src.utils.config_loader import settings

logger = logging.getLogger(__name__)

_client = None


class _NoOpSpan:
    """
    No-op stand-in for a Langfuse trace/span/generation.

    Used only when Langfuse tracing is explicitly disabled.

    We intentionally do NOT silently fall back to this object when
    Langfuse is enabled but initialization fails. If tracing is enabled,
    a configuration/connectivity problem should be visible.
    """

    def span(self, **kwargs):
        return _NoOpSpan()

    def generation(self, **kwargs):
        return _NoOpSpan()

    def event(self, **kwargs):
        return _NoOpSpan()

    def update(self, **kwargs):
        return self

    def end(self, **kwargs):
        return None


class _NoOpTracer:
    """
    No-op stand-in for the Langfuse client.

    This is used only when Langfuse is explicitly disabled in config.
    """

    def trace(self, **kwargs):
        return _NoOpSpan()

    def flush(self):
        return None

    def shutdown(self):
        return None


def _normalize_host(host: str) -> str:
    """
    Normalize Langfuse host before passing it to the SDK.

    Examples:

        https://cloud.langfuse.com
        https://cloud.langfuse.com/
    """

    if not host:
        raise ValueError("LANGFUSE_HOST is empty.")

    host = host.strip().rstrip("/")

    if not host.startswith(("http://", "https://")):
        raise ValueError(
            "LANGFUSE_HOST must include the URL scheme, "
            "for example https://cloud.langfuse.com"
        )

    return host


def _validate_credentials(public_key: str, secret_key: str, host: str) -> None:
    """
    Validate Langfuse configuration without logging secret values.
    """

    if not public_key or not public_key.strip():
        raise ValueError("Langfuse public key is empty.")

    if not secret_key or not secret_key.strip():
        raise ValueError("Langfuse secret key is empty.")

    if not host or not host.strip():
        raise ValueError("Langfuse host is empty.")


def get_tracer():
    """
    Return the singleton Langfuse client.

    Behavior:

    1. If Langfuse is disabled, return the no-op client.
    2. If enabled, initialize the real Langfuse client.
    3. Validate credentials and host.
    4. Run auth_check() to verify connectivity.
    5. Fail loudly if initialization/authentication fails.

    We deliberately do not silently convert an enabled Langfuse
    configuration into a NoOpTracer. Doing so makes production
    observability failures invisible.
    """

    global _client

    if _client is not None:
        return _client

    if not settings.langfuse_enabled:
        logger.info("Langfuse tracing is disabled by configuration.")
        _client = _NoOpTracer()
        return _client

    public_key = (settings.secrets.langfuse_public_key or "").strip()
    secret_key = (settings.secrets.langfuse_secret_key or "").strip()
    host = _normalize_host(settings.secrets.langfuse_host)

    _validate_credentials(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )

    logger.info(
        "Initializing Langfuse client: host=%s, public_key_present=%s, "
        "secret_key_present=%s",
        host,
        bool(public_key),
        bool(secret_key),
    )

    try:
        from langfuse import Langfuse

        # Keep the existing Langfuse v2 API.
        #
        # flush_at=1 and flush_interval=1 make tracing deterministic
        # for this application while debugging. The application is
        # Streamlit-based and the amount of trace data per turn is small.
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            flush_at=1,
            flush_interval=1,
            debug=True,
        )

        logger.info(
            "Langfuse client created successfully. "
            "Running authentication/connectivity check..."
        )

        # Langfuse recommends auth_check() during setup/troubleshooting.
        auth_ok = _client.auth_check()

        if not auth_ok:
            raise RuntimeError(
                "Langfuse auth_check() returned False. "
                "Verify Langfuse credentials, host and network connectivity."
            )

        logger.info(
            "Langfuse authentication/connectivity check succeeded."
        )

        return _client

    except Exception:
        logger.exception(
            "Langfuse initialization/authentication failed. "
            "Tracing is enabled but the Langfuse client could not be "
            "initialized successfully."
        )

        # Important:
        #
        # Do NOT silently replace the client with _NoOpTracer().
        #
        # If LANGFUSE is enabled, we want the actual problem to be visible
        # in the Databricks App logs.
        _client = None

        raise


@contextmanager
def trace_span(parent, name, **kwargs):
    """
    Create a Langfuse span under a trace/span.

    The existing application code uses this helper throughout the graph,
    so keeping this API unchanged means no other application files need
    to be modified.

    Example:

        with trace_span(trace, "sql_execution", input=sql) as span:
            result = execute(sql)
            span.update(output=result)

    If the wrapped operation raises an exception, mark the Langfuse
    span as ERROR and re-raise the original exception.
    """

    span = parent.span(name=name, **kwargs)

    try:
        yield span

    except Exception as exc:
        try:
            span.update(
                level="ERROR",
                status_message=str(exc),
            )
        except Exception:
            logger.exception(
                "Failed to update Langfuse span '%s' after exception.",
                name,
            )

        raise

    finally:
        try:
            span.end()
        except Exception:
            logger.exception(
                "Failed to end Langfuse span '%s'.",
                name,
            )


def flush():
    """
    Flush all queued Langfuse observations.

    This is intentionally exposed as a module-level helper so callers
    do not need to know which concrete Langfuse client implementation
    is being used.
    """

    global _client

    if _client is None:
        return

    try:
        _client.flush()
        logger.debug("Langfuse flush completed successfully.")
    except Exception:
        logger.exception("Langfuse flush failed.")


def shutdown():
    """
    Gracefully shut down Langfuse.

    Primarily useful when the application process is terminating.
    """

    global _client

    if _client is None:
        return

    try:
        if hasattr(_client, "shutdown"):
            _client.shutdown()
        else:
            _client.flush()

        logger.info("Langfuse client shutdown completed.")

    except Exception:
        logger.exception("Langfuse shutdown failed.")


def reset_tracer():
    """
    Reset the cached client.

    This is primarily useful for tests or application reloads.

    It does not modify Langfuse configuration or credentials.
    """

    global _client
    _client = None