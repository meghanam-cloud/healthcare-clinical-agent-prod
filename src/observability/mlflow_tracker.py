"""
MLflow 3 experiment logging for every agent turn: the system prompt used,
the generated SQL, the user's question, the final response and the LLM
endpoint that produced it. One MLflow run per conversation turn.
"""

import logging
import mlflow
from src.utils.config_loader import settings

logger = logging.getLogger(__name__)

_experiment_ready = False


def _ensure_experiment():
    global _experiment_ready
    if _experiment_ready:
        return
    try:
        mlflow.set_experiment(settings.mlflow_experiment_path)
    except Exception as e:
        logger.warning("Could not set MLflow experiment %s: %s", settings.mlflow_experiment_path, e)
    _experiment_ready = True


def log_turn(
    question: str,
    system_prompt: str,
    sql_query: str,
    response: str,
    model_name: str,
    latency_seconds: float = None,
    sql_row_count: int = None,
    guardrail_blocked: bool = False,
    error: str = None,
):
    """Log one chat turn as an MLflow run. Never raises - logging failures
    must not break the user-facing chat response."""
    _ensure_experiment()
    try:
        with mlflow.start_run(run_name="sql_agent_turn"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("guardrail_blocked", guardrail_blocked)
            mlflow.log_text(question or "", "question.txt")
            mlflow.log_text(system_prompt or "", "system_prompt.txt")
            mlflow.log_text(sql_query or "", "generated_sql.sql")
            mlflow.log_text(response or "", "response.txt")
            if latency_seconds is not None:
                mlflow.log_metric("latency_seconds", latency_seconds)
            if sql_row_count is not None:
                mlflow.log_metric("sql_row_count", sql_row_count)
            if error:
                mlflow.log_param("error", str(error)[:250])
    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)
