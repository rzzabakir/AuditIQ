"""Tests for engine.checks.distribution."""
import pandas as pd
import pytest

from engine.checks.distribution import outliers, value_concentration


# ── outliers ──────────────────────────────────────────────────


def test_outliers_no_outliers():
    df = pd.DataFrame({"score": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]})
    assert outliers(df) == []


def test_outliers_detects_extreme_value():
    # 9 tightly grouped values + one extreme outlier
    df = pd.DataFrame({"score": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
    results = outliers(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "outliers"
    assert r["column"] == "score"
    assert r["count"] == 1
    assert r["severity"] == "warn"


def test_outliers_skips_below_min_values():
    # Fewer than 10 non-null values — should be skipped
    df = pd.DataFrame({"score": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    assert outliers(df) == []


def test_outliers_zero_iqr_skipped():
    # All identical values — IQR is 0, should be skipped
    df = pd.DataFrame({"score": [5] * 15})
    assert outliers(df) == []


def test_outliers_skips_non_numeric():
    df = pd.DataFrame({"city": ["London"] * 15})
    assert outliers(df) == []


def test_outliers_handles_nulls_gracefully():
    data = [10, 11, 12, 13, 14, 15, 16, 17, 18, None, 1000]
    df = pd.DataFrame({"score": data})
    results = outliers(df)
    assert len(results) == 1
    assert results[0]["count"] == 1


def test_outliers_includes_fence_in_notes():
    df = pd.DataFrame({"score": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 500]})
    r = outliers(df)[0]
    assert "Q1=" in r["notes"]
    assert "IQR=" in r["notes"]


# ── value_concentration ───────────────────────────────────────


def test_value_concentration_no_issue():
    df = pd.DataFrame({"status": ["active", "inactive", "pending", "active", "closed"]})
    assert value_concentration(df) == []


def test_value_concentration_fires_on_dominant_value():
    df = pd.DataFrame({"status": ["active"] * 95 + ["inactive"] * 5})
    results = value_concentration(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "value_concentration"
    assert r["severity"] == "info"


def test_value_concentration_exactly_at_threshold():
    # 90 out of 100 = exactly 90% — should fire at default threshold of 0.90
    df = pd.DataFrame({"col": ["A"] * 90 + ["B"] * 10})
    results = value_concentration(df)
    assert len(results) == 1


def test_value_concentration_custom_threshold():
    df = pd.DataFrame({"col": ["A"] * 80 + ["B"] * 20})
    assert value_concentration(df) == []
    assert len(value_concentration(df, threshold=0.75)) == 1


def test_value_concentration_empty_dataframe():
    df = pd.DataFrame({"col": []})
    assert value_concentration(df) == []


def test_value_concentration_all_null():
    df = pd.DataFrame({"col": [None, None, None]})
    assert value_concentration(df) == []
