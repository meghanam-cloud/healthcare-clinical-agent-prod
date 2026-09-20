"""
Node: builds the schema context string injected into the SQL generation
prompt. Combines:
  1. Live column list/types from DESCRIBE TABLE (source of truth, catches
     schema drift), with a safe fallback to the static config.yaml mapping
     if the warehouse call fails (e.g. running offline/unit tests).
  2. Hand-written semantic notes on what each column means and how the
     semi-structured (JSON-as-string) columns are shaped, so the LLM knows
     HOW to query them (e.g. get_json_object) not just that they exist.
"""

import logging
from src.agent.state import AgentState
from src.utils.config_loader import settings
from src.utils.sql_connection import run_query
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)

# Notes about what each column contains, to ground the SQL generator's
# understanding of the clinical cohort table.
COLUMN_NOTES = {
    "patient_id": "Unique patient identifier, STRING (e.g. 'PT-2000'). Use for joins/dedup.",
    "age": "Patient age in years, INT.",
    "gender": "Patient gender, STRING.",
    "race_ethnicity": "Patient race/ethnicity, STRING.",
    "primary_condition": "Primary chronic condition, STRING (e.g. 'Type 2 Diabetes', 'Hypertension', 'CKD Stage 3', 'Heart Failure', 'Asthma', 'COPD'). Use LIKE/ILIKE or exact match for filtering by condition.",
    "bmi": "Body mass index, DOUBLE.",
    "hba1c_level": "Glycated hemoglobin percentage, DOUBLE. >8.5 is the uncontrolled/high-risk threshold per clinical guidelines; 7.0-8.5 is moderate; <7.0 is well-controlled.",
    "blood_pressure": "Blood pressure reading as free-text STRING 'systolic/diastolic' (e.g. '160/109'). Split on '/' to compare systolic/diastolic separately; do not cast the whole string to a number.",
    "egfr_ml_min": "Estimated glomerular filtration rate (renal function), INT, mL/min. <45 is high-risk/CKD-relevant; 45-59 is moderate; >=60 is normal.",
    "ldl_mg_dl": "LDL cholesterol, INT, mg/dL.",
    "smoking_status": "Smoking status, STRING ('Never', 'Former', 'Current').",
    "known_allergies": "Free-text STRING of known drug allergies, or 'NKDA' (no known drug allergies). Use LIKE for keyword search (e.g. '%Penicillin%').",
    "current_medications": "Free-text STRING, comma-separated list of active medications with dose/frequency (e.g. 'Lisinopril 20mg QD, Amlodipine 5mg QD'). Use LIKE for keyword search on a drug name.",
    "med_adherence_rate": "Medication adherence score, DOUBLE 0-1 (e.g. 0.91 = 91%). >=0.80 optimal, 0.50-0.79 suboptimal, <0.50 critical non-adherence.",
    "ed_visits_12m": "Emergency department visits in the trailing 12 months, INT.",
    "hospitalizations_12m": "Hospitalizations in the trailing 12 months, INT.",
    "risk_tier": "Pre-computed risk tier, STRING ('Low', 'Moderate', 'High').",
    "readmission_30d_prob": "30-day readmission probability, DOUBLE 0-1. >0.65 flags enrollment in the Post-Discharge Care Transition Program.",
    "last_encounter_date": "Date of the most recent clinical encounter, DATE.",
    "primary_care_provider": "Assigned primary care provider name, STRING.",
}


def _describe_table_live() -> list[str]:
    columns, rows = run_query(f"DESCRIBE TABLE {settings.data.full_table_name}")
    lines = []
    for row in rows:
        col_name, col_type = row[0], row[1]
        if not col_name or col_name.startswith("#"):
            continue
        note = COLUMN_NOTES.get(col_name, "")
        lines.append(f"  - {col_name} ({col_type}): {note}")
    return lines


def _describe_table_static() -> list[str]:
    lines = []
    for logical_name, physical_col in settings.data.columns.items():
        note = COLUMN_NOTES.get(physical_col, "")
        lines.append(f"  - {physical_col} [config alias: {logical_name}]: {note}")
    return lines


def build_schema_context() -> str:
    try:
        column_lines = _describe_table_live()
        source = "live DESCRIBE TABLE"
    except Exception as e:
        logger.warning("DESCRIBE TABLE failed, using static config schema: %s", e)
        column_lines = _describe_table_static()
        source = "static config.yaml mapping"

    return (
        f"Table: {settings.data.full_table_name}\n"
        f"(schema source: {source})\n"
        f"Columns:\n" + "\n".join(column_lines) + "\n\n"
        "Notes:\n"
        "  - Always filter/search using the exact column names above.\n"
        "  - Numeric clinical fields (hba1c_level, egfr_ml_min, bmi, ldl_mg_dl, "
        "med_adherence_rate, readmission_30d_prob) are numeric; do not quote them.\n"
        "  - blood_pressure is a 'systolic/diastolic' STRING - use "
        "split(blood_pressure, '/')[0] / [1] cast to INT to compare against a threshold.\n"
        "  - This table contains only de-identified cohort demo data for care-coordination "
        "queries (risk tiers, adherence, labs, medications on file) - never fabricate a "
        "diagnosis or treatment recommendation from a SQL result alone.\n"
        "  - Always add a LIMIT (default 50) unless the user asks for a count/aggregate."
    )


def schema_retriever_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    with trace_span(
        trace,
        "schema_retrieval",
        input={"table": settings.data.full_table_name},
    ) as span:
        schema_context = build_schema_context()
        span.update(
            output=schema_context[:2000],
            metadata={"full_length": len(schema_context)},
        )

    return {**state, "schema_context": schema_context}
