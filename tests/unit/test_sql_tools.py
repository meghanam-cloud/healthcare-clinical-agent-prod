from src.agent.tools.sql_tools import _is_safe_select


HEALTHCARE_TABLE = "healthcare_patient_tbl"


def test_rejects_non_select():
    ok, _ = _is_safe_select(
        f"DELETE FROM {HEALTHCARE_TABLE}"
    )
    assert not ok


def test_rejects_multi_statement():
    ok, _ = _is_safe_select(
        f"SELECT * FROM {HEALTHCARE_TABLE}; "
        f"DROP TABLE {HEALTHCARE_TABLE}"
    )
    assert not ok


def test_rejects_wrong_table():
    ok, _ = _is_safe_select(
        "SELECT * FROM some_other_table"
    )
    assert not ok


def test_accepts_valid_select():
    ok, reason = _is_safe_select(
        f"""
        SELECT
            primary_condition,
            COUNT(*) AS patient_count
        FROM {HEALTHCARE_TABLE}
        WHERE primary_condition = 'diabetes'
        """
    )
    assert ok, reason


def test_accepts_healthcare_patient_columns():
    ok, reason = _is_safe_select(
        f"""
        SELECT
            patient_id,
            age,
            primary_condition,
            hba1c_level,
            risk_tier
        FROM {HEALTHCARE_TABLE}
        WHERE risk_tier = 'high'
        """
    )
    assert ok, reason


def test_rejects_insert():
    ok, _ = _is_safe_select(
        f"INSERT INTO {HEALTHCARE_TABLE} "
        "(patient_id) VALUES ('123')"
    )
    assert not ok


def test_rejects_update():
    ok, _ = _is_safe_select(
        f"UPDATE {HEALTHCARE_TABLE} "
        "SET risk_tier = 'high'"
    )
    assert not ok


def test_rejects_drop():
    ok, _ = _is_safe_select(
        f"DROP TABLE {HEALTHCARE_TABLE}"
    )
    assert not ok


def test_rejects_truncate():
    ok, _ = _is_safe_select(
        f"TRUNCATE TABLE {HEALTHCARE_TABLE}"
    )
    assert not ok


def test_rejects_alter():
    ok, _ = _is_safe_select(
        f"ALTER TABLE {HEALTHCARE_TABLE} ADD COLUMN foo STRING"
    )
    assert not ok


def test_rejects_grant():
    ok, _ = _is_safe_select(
        f"GRANT SELECT ON TABLE {HEALTHCARE_TABLE} TO some_user"
    )
    assert not ok