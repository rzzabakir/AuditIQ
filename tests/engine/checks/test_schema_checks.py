"""Tests for engine.checks.schema_checks."""
import pandas as pd
import pytest

from engine.checks.schema_checks import run_schema_checks


def _schema(*cols):
    """Quick helper: build a schema dict from (name, type) tuples."""
    return {name: {"expected_type": typ} for name, typ in cols}


# ── missing / unexpected columns ─────────────────────────────


def test_missing_required_column():
    schema = {"customer_id": {"expected_type": "identifier"}}
    df = pd.DataFrame({"name": ["Alice"]})
    results = run_schema_checks(schema, df)
    checks = [r["check"] for r in results]
    assert "schema_missing_column" in checks
    missing = next(r for r in results if r["check"] == "schema_missing_column")
    assert missing["column"] == "customer_id"
    assert missing["severity"] == "bad"


def test_unexpected_column_flagged():
    schema = {"name": {"expected_type": "text"}}
    df = pd.DataFrame({"name": ["Alice"], "extra_col": [1]})
    results = run_schema_checks(schema, df)
    checks = [r["check"] for r in results]
    assert "schema_unexpected_columns" in checks
    extra = next(r for r in results if r["check"] == "schema_unexpected_columns")
    assert extra["column"] == "extra_col"
    assert extra["severity"] == "info"


def test_empty_schema_returns_nothing():
    df = pd.DataFrame({"col": [1, 2, 3]})
    assert run_schema_checks({}, df) == []


# ── type mismatch ─────────────────────────────────────────────


def test_type_mismatch_numeric_with_text_values():
    schema = {"amount": {"expected_type": "numeric"}}
    df = pd.DataFrame({"amount": ["100", "n/a", "250"]})
    results = run_schema_checks(schema, df)
    mismatch = next((r for r in results if r["check"] == "schema_type_mismatch_numeric"), None)
    assert mismatch is not None
    assert mismatch["count"] == 1  # "n/a" cannot be coerced


def test_type_mismatch_date_with_non_dates():
    schema = {"dob": {"expected_type": "date"}}
    df = pd.DataFrame({"dob": ["1990-01-15", "not-a-date", "1985-07-20"]})
    results = run_schema_checks(schema, df)
    mismatch = next((r for r in results if r["check"] == "schema_type_mismatch_date"), None)
    assert mismatch is not None
    assert mismatch["count"] == 1


def test_type_mismatch_text_with_numeric():
    schema = {"notes": {"expected_type": "text"}}
    df = pd.DataFrame({"notes": ["memo text", "42", "another note"]})
    results = run_schema_checks(schema, df)
    mismatch = next((r for r in results if r["check"] == "schema_type_mismatch_text"), None)
    assert mismatch is not None
    assert mismatch["count"] == 1


def test_no_type_mismatch_on_clean_data():
    schema = {"amount": {"expected_type": "numeric"}}
    df = pd.DataFrame({"amount": [100, 200, 300]})
    results = run_schema_checks(schema, df)
    mismatches = [r for r in results if "mismatch" in r["check"]]
    assert mismatches == []


# ── range violations ──────────────────────────────────────────


def test_schema_below_min():
    schema = {"age": {"expected_type": "numeric", "min_value": 18}}
    df = pd.DataFrame({"age": [25, 16, 30]})
    results = run_schema_checks(schema, df)
    below = next((r for r in results if r["check"] == "schema_below_min"), None)
    assert below is not None
    assert below["count"] == 1


def test_schema_above_max():
    schema = {"score": {"expected_type": "numeric", "max_value": 100}}
    df = pd.DataFrame({"score": [80, 105, 95]})
    results = run_schema_checks(schema, df)
    above = next((r for r in results if r["check"] == "schema_above_max"), None)
    assert above is not None
    assert above["count"] == 1


def test_schema_range_no_violation():
    schema = {"score": {"expected_type": "numeric", "min_value": 0, "max_value": 100}}
    df = pd.DataFrame({"score": [50, 75, 100]})
    results = run_schema_checks(schema, df)
    range_issues = [r for r in results if r["check"] in ("schema_below_min", "schema_above_max")]
    assert range_issues == []


# ── format violations ─────────────────────────────────────────


def test_schema_format_violation():
    # Simple 5-digit zip code pattern — one clear violation
    schema = {"zipcode": {"allowed_formats": r"\d{5}"}}
    df = pd.DataFrame({"zipcode": ["10001", "INVALID", "90210"]})
    results = run_schema_checks(schema, df)
    violation = next((r for r in results if r["check"] == "schema_format_violation"), None)
    assert violation is not None
    assert violation["count"] == 1


def test_schema_format_all_match():
    schema = {"code": {"allowed_formats": r"\d{3}"}}
    df = pd.DataFrame({"code": ["001", "042", "999"]})
    results = run_schema_checks(schema, df)
    violations = [r for r in results if r["check"] == "schema_format_violation"]
    assert violations == []


def test_schema_invalid_regex_gracefully_skipped():
    schema = {"col": {"allowed_formats": r"[unclosed"}}
    df = pd.DataFrame({"col": ["abc", "def"]})
    # Should not raise — invalid regex is logged and skipped
    results = run_schema_checks(schema, df)
    violations = [r for r in results if r["check"] == "schema_format_violation"]
    assert violations == []
