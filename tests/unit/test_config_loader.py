from src.utils.config_loader import settings


def test_data_config_loaded():
    assert settings.data.table == "healthcare_patient_tbl"
    assert settings.data.full_table_name.endswith(".healthcare_patient_tbl")


def test_llm_config_loaded():
    assert settings.llm.endpoint_name == "databricks-meta-llama-3-3-70b-instruct"


def test_guardrails_config_loaded():
    assert settings.guardrails.pii_check is True
    assert settings.guardrails.prompt_injection_check is True
    assert settings.guardrails.topical_scope_check is True

    # Healthcare-domain keywords
    assert "patient" in settings.guardrails.allowed_topics_keywords
    assert "diagnosis" in settings.guardrails.allowed_topics_keywords
    assert "medication" in settings.guardrails.allowed_topics_keywords
    assert "guideline" in settings.guardrails.allowed_topics_keywords