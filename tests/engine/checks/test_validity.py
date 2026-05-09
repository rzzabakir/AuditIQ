"""Tests for engine.checks.validity."""
import pandas as pd
import pytest

from engine.checks.validity import invalid_dates, malformed_identifiers, mixed_types


# ── mixed_types ───────────────────────────────────────────────


def test_mixed_types_uniform_string():
    df = pd.DataFrame({"city": ["London", "Paris", "Berlin"]})
    assert mixed_types(df) == []


def test_mixed_types_uniform_numeric_dtype():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    assert mixed_types(df) == []


def test_mixed_types_detects_mixed_string_and_int():
    df = pd.DataFrame({"col": ["abc", "123", "def", "456", "ghi", "789",
                                "jkl", "012", "mno", "pqr"]})
    results = mixed_types(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "mixed_types"
    assert r["column"] == "col"
    assert r["severity"] == "warn"
    assert r["type_counts"]["int"] > 0
    assert r["type_counts"]["string"] > 0


def test_mixed_types_minority_below_threshold_no_fire():
    # threshold is < 0.05 (strict), so 1/21 ≈ 4.76% is below and should not fire
    data = ["text"] * 20 + ["42"]
    df = pd.DataFrame({"col": data})
    assert mixed_types(df) == []


def test_mixed_types_empty_column():
    df = pd.DataFrame({"col": [None, None]})
    assert mixed_types(df) == []


def test_mixed_types_skips_non_object_dtype():
    df = pd.DataFrame({"col": pd.array([1, 2, 3], dtype="int64")})
    assert mixed_types(df) == []


# ── invalid_dates ─────────────────────────────────────────────


def test_invalid_dates_valid_column():
    df = pd.DataFrame({"date_of_birth": ["1990-01-15", "1985-07-20", "2000-03-05"]})
    assert invalid_dates(df) == []


def test_invalid_dates_non_date_column_ignored():
    df = pd.DataFrame({"city": ["London", "Paris", "not-a-date"]})
    assert invalid_dates(df) == []


def test_invalid_dates_fires_on_mostly_unparseable():
    df = pd.DataFrame({"signup_date": ["not", "a", "date", "at", "all",
                                        "gibberish", "foo", "bar", "baz", "qux"]})
    results = invalid_dates(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "invalid_dates"
    assert r["column"] == "signup_date"


def test_invalid_dates_already_datetime_dtype_skipped():
    df = pd.DataFrame({"created_date": pd.to_datetime(["2020-01-01", "2021-06-15"])})
    assert invalid_dates(df) == []


def test_invalid_dates_all_null_skipped():
    df = pd.DataFrame({"submitted_date": [None, None, None]})
    assert invalid_dates(df) == []


# ── malformed_identifiers ─────────────────────────────────────


def test_malformed_identifiers_valid_emails():
    df = pd.DataFrame({"email": ["a@b.com", "user@example.org", "x@y.co.uk"]})
    assert malformed_identifiers(df) == []


def test_malformed_identifiers_bad_emails():
    df = pd.DataFrame({"email": ["valid@email.com", "not-an-email", "also-bad"]})
    results = malformed_identifiers(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "malformed_identifiers"
    assert r["count"] == 2


def test_malformed_identifiers_valid_phones():
    df = pd.DataFrame({"phone": ["555-010-1234", "+1-800-555-0100", "07911123456"]})
    assert malformed_identifiers(df) == []


def test_malformed_identifiers_bad_phones():
    df = pd.DataFrame({"phone": ["555", "1234567890123456"]})  # too short, too long
    results = malformed_identifiers(df)
    assert len(results) == 1
    assert results[0]["count"] == 2


def test_malformed_identifiers_non_identifier_column_skipped():
    df = pd.DataFrame({"city": ["London", "not-an-email", "Paris"]})
    assert malformed_identifiers(df) == []


def test_malformed_identifiers_non_object_dtype_skipped():
    df = pd.DataFrame({"phone": [5550101, 5550102]})
    assert malformed_identifiers(df) == []
