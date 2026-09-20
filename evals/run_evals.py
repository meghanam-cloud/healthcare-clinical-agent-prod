"""Golden-question evaluation for the healthcare agentic RAG graph."""

import json
import logging
import uuid

import mlflow

from src.agent.graph import run_agent
from src.utils.config_loader import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(path: str = "evals/eval_dataset.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_case(case: dict) -> dict:
    session_id = str(uuid.uuid4())
    result = run_agent(
        question=case["question"],
        session_id=session_id,
        user_id="eval-runner",
        chat_history=[],
    )

    if case.get("expect_blocked"):
        passed = bool(result.get("blocked"))
    else:
        plan = result.get("plan", {}) or {}
        actual_tools = [step.get("tool") for step in plan.get("steps", [])]
        expected_tools = case.get("expected_tools", [])
        tools_ok = set(actual_tools) == set(expected_tools) if expected_tools else True

        sql = (result.get("sql_query") or "").lower()
        expected_sql = case.get("expected_sql_contains", [])
        sql_ok = all(term.lower() in sql for term in expected_sql)

        answer = (result.get("final_response") or "").lower()
        expected_answer = case.get("expected_answer_contains", [])
        answer_ok = all(term.lower() in answer for term in expected_answer)

        passed = (
            tools_ok
            and sql_ok
            and answer_ok
            and bool(result.get("final_response"))
            and bool(result.get("review_passed"))
        )

    return {
        "question": case["question"],
        "passed": passed,
        "tools": [s.get("tool") for s in (result.get("plan", {}) or {}).get("steps", [])],
        "plan": result.get("plan", {}),
        "review_attempts": result.get("review_attempts", 0),
        "review_decision": result.get("review_decision"),
        "sql_query": result.get("sql_query", ""),
        "blocked": result.get("blocked", False),
        "response": result.get("final_response", ""),
    }


def main():
    mlflow.set_experiment(settings.mlflow_experiment_path)
    dataset = load_dataset()

    with mlflow.start_run(run_name="clinical_agentic_eval_suite"):
        results = [evaluate_case(c) for c in dataset]
        pass_count = sum(r["passed"] for r in results)

        mlflow.log_metric("total_cases", len(results))
        mlflow.log_metric("passed_cases", pass_count)
        mlflow.log_metric("pass_rate", pass_count / len(results) if results else 0)
        mlflow.log_text(json.dumps(results, indent=2), "eval_results.json")

        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            logger.info(
                "[%s] %s | tools=%s | review_attempts=%s",
                status,
                r["question"],
                r["tools"],
                r["review_attempts"],
            )

        print(f"\n{pass_count}/{len(results)} cases passed.")


if __name__ == "__main__":
    main()
