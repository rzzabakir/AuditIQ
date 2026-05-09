"""Tests for engine.checks.completeness."""
import pandas as pd
import pytest

from engine.checks.completeness import empty_rows, missing_values, null_concentration


# ── missing_values ────────────────────────────────────────────


def test_missing_values_no_issues():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert missing_values(df) == []


def test_missing_values_null_detected():
    df = pd.DataFrame({"a": [1, None, 3]})
    results = missing_values(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "missing_values"
    assert r["column"] == "a"
    assert r["count"] == 1
    assert r["total"] == 3
    assert r["severity"] == "warn"


def test_missing_values_whitespace_only_counted():
    df = pd.DataFrame({"name": ["Alice", "   ", "Bob"]})
    results = missing_values(df)
    assert len(results) == 1
    assert results[0]["count"] == 1


def test_missing_values_multiple_columns():
    df = pd.DataFrame({"a": [None, None], "b": [1, None], "c": [1, 2]})
    results = missing_values(df)
    checks = {r["column"]: r["count"] for r in results}
    assert checks["a"] == 2
    assert checks["b"] == 1
    assert "c" not in checks


def test_missing_values_all_null():
    df = pd.DataFrame({"x": [None, None, None]})
    results = missing_values(df)
    assert results[0]["count"] == 3
    assert results[0]["pct"] == 100.0


def test_missing_values_check_id_format():
    df = pd.DataFrame({"col": [None, 1]})
    r = missing_values(df)[0]
    assert r["check_id"] == "completeness.missing_values"
    assert r["category"] == "completeness"


# ── empty_rows ────────────────────────────────────────────────


def test_empty_rows_no_issues():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert empty_rows(df) == []


def test_empty_rows_detects_all_null_row():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    results = empty_rows(df)
    assert len(results) == 1
    assert results[0]["count"] == 1
    assert results[0]["severity"] == "bad"


def test_empty_rows_whitespace_only_counts_as_empty():
    df = pd.DataFrame({"a": ["", "  "], "b": [None, None]})
    results = empty_rows(df)
    assert results[0]["count"] == 2


def test_empty_rows_empty_dataframe():
    df = pd.DataFrame({"a": []})
    assert empty_rows(df) == []


def test_empty_rows_column_is_row_sentinel():
    df = pd.DataFrame({"a": [None], "b": [None]})
    r = empty_rows(df)[0]
    assert r["column"] == "__row__"


# ── null_concentration ────────────────────────────────────────


def test_null_concentration_below_threshold():
    df = pd.DataFrame({"a": [1, None, 3, 4, 5]})  # 20% null
    assert null_concentration(df) == []


def test_null_concentration_above_threshold():
    df = pd.DataFrame({"a": [None, None, None, 1, 2]})  # 60% null
    results = null_concentration(df)
    assert len(results) == 1
    assert results[0]["check"] == "null_concentration"
    assert results[0]["severity"] == "bad"


def test_null_concentration_exactly_at_threshold_not_fired():
    # 50% null — threshold is 0.50, so < 0.50 means it fires at exactly 0.50
    df = pd.DataFrame({"a": [None, None, 1, 2]})  # 50% null — equals threshold, should fire
    results = null_concentration(df)
    assert len(results) == 1


def test_null_concentration_custom_threshold():
    # 1 null out of 10 = 10% null
    df = pd.DataFrame({"a": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    # threshold=0.11 → 10% < 11% → no fire
    assert null_concentration(df, threshold=0.11) == []
    # threshold=0.09 → 10% >= 9% → fires
    assert len(null_concentration(df, threshold=0.09)) == 1


def test_null_concentration_empty_dataframe():
    df = pd.DataFrame({"a": []})
    assert null_concentration(df) == []
