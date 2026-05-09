"""Tests for engine.checks.format_checks."""
import pandas as pd
import pytest

from engine.checks.format_checks import (
    casing_inconsistency,
    format_inconsistencies,
    whitespace_issues,
)


# ── format_inconsistencies ────────────────────────────────────


def test_format_inconsistencies_clean_column():
    df = pd.DataFrame({"city": ["London", "Paris", "Berlin"]})
    assert format_inconsistencies(df) == []


def test_format_inconsistencies_column_name_as_value():
    df = pd.DataFrame({"status": ["active", "inactive", "status"]})
    results = format_inconsistencies(df)
    checks = [r["check"] for r in results]
    assert "format_inconsistency_name_as_value" in checks
    name_result = next(r for r in results if r["check"] == "format_inconsistency_name_as_value")
    assert name_result["count"] == 1


def test_format_inconsistencies_numeric_as_text():
    df = pd.DataFrame({"notes": ["some text", "42", "more text", "100", "other"]})
    results = format_inconsistencies(df)
    checks = [r["check"] for r in results]
    assert "format_inconsistency_numeric_as_text" in checks
    num_result = next(r for r in results if r["check"] == "format_inconsistency_numeric_as_text")
    assert num_result["count"] == 2


def test_format_inconsistencies_skips_non_object_dtype():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    assert format_inconsistencies(df) == []


def test_format_inconsistencies_skips_all_null():
    df = pd.DataFrame({"col": [None, None]})
    assert format_inconsistencies(df) == []


# ── casing_inconsistency ──────────────────────────────────────


def test_casing_inconsistency_uniform_lowercase():
    df = pd.DataFrame({"country": ["usa", "uk", "germany"]})
    assert casing_inconsistency(df) == []


def test_casing_inconsistency_uniform_titlecase():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})
    assert casing_inconsistency(df) == []


def test_casing_inconsistency_detects_mixed():
    df = pd.DataFrame({"email": [
        "alice@example.com",
        "bob@example.com",
        "CAROL@EXAMPLE.COM",
        "dave@example.com",
        "EVE@EXAMPLE.COM",
    ]})
    results = casing_inconsistency(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "casing_inconsistency"
    assert r["severity"] == "warn"


def test_casing_inconsistency_dominant_style_over_95pct_no_fire():
    # 19 lowercase + 1 uppercase = 95% dominant → should not fire
    data = ["lower"] * 19 + ["UPPER"]
    df = pd.DataFrame({"col": data})
    assert casing_inconsistency(df) == []


def test_casing_inconsistency_empty_column():
    df = pd.DataFrame({"col": pd.Series([], dtype=object)})
    assert casing_inconsistency(df) == []


def test_casing_inconsistency_skips_non_object():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    assert casing_inconsistency(df) == []


# ── whitespace_issues ─────────────────────────────────────────


def test_whitespace_issues_clean():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})
    assert whitespace_issues(df) == []


def test_whitespace_issues_leading_space():
    df = pd.DataFrame({"name": [" Alice", "Bob"]})
    results = whitespace_issues(df)
    assert len(results) == 1
    assert results[0]["count"] == 1
    assert results[0]["severity"] == "info"


def test_whitespace_issues_trailing_space():
    df = pd.DataFrame({"name": ["Alice ", "Bob"]})
    results = whitespace_issues(df)
    assert results[0]["count"] == 1


def test_whitespace_issues_multiple_columns():
    df = pd.DataFrame({
        "first": [" Alice", "Bob"],
        "last":  ["Smith ", "Jones "],
    })
    results = whitespace_issues(df)
    cols = {r["column"] for r in results}
    assert "first" in cols
    assert "last" in cols


def test_whitespace_issues_skips_non_object():
    df = pd.DataFrame({"amount": [1.0, 2.0]})
    assert whitespace_issues(df) == []
