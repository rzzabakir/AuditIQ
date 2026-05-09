"""Tests for engine.checks.range_checks."""
import pandas as pd
import pytest
from datetime import datetime, timedelta

from engine.checks.range_checks import future_dates, impossible_values


# ── future_dates ──────────────────────────────────────────────


def test_future_dates_all_past():
    df = pd.DataFrame({"date_of_birth": ["1990-01-01", "1985-06-15", "2000-03-22"]})
    assert future_dates(df) == []


def test_future_dates_detects_future_value():
    future = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
    df = pd.DataFrame({"signup_date": ["2020-01-01", future]})
    results = future_dates(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "future_dates"
    assert r["count"] == 1
    assert r["severity"] == "warn"


def test_future_dates_non_date_column_skipped():
    future = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
    df = pd.DataFrame({"city": ["London", future]})
    assert future_dates(df) == []


def test_future_dates_all_future():
    future1 = (datetime.now() + timedelta(days=100)).strftime("%Y-%m-%d")
    future2 = (datetime.now() + timedelta(days=200)).strftime("%Y-%m-%d")
    df = pd.DataFrame({"submitted_date": [future1, future2]})
    results = future_dates(df)
    assert results[0]["count"] == 2


def test_future_dates_all_null_skipped():
    df = pd.DataFrame({"created_date": [None, None]})
    assert future_dates(df) == []


# ── impossible_values ─────────────────────────────────────────


def test_impossible_values_valid_ages():
    df = pd.DataFrame({"age": [25, 40, 65, 18]})
    assert impossible_values(df) == []


def test_impossible_values_negative_age():
    df = pd.DataFrame({"age": [25, -5, 0]})
    results = impossible_values(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "impossible_values"
    assert r["count"] == 2  # -5 and 0 both invalid (age must be 1–120)
    assert r["severity"] == "bad"


def test_impossible_values_age_over_120():
    df = pd.DataFrame({"age": [30, 121, 150]})
    results = impossible_values(df)
    assert results[0]["count"] == 2


def test_impossible_values_valid_amounts():
    df = pd.DataFrame({"purchase_amount": [10.0, 250.0, 0.0]})
    assert impossible_values(df) == []


def test_impossible_values_negative_amount():
    df = pd.DataFrame({"purchase_amount": [100.0, -50.0, 200.0]})
    results = impossible_values(df)
    assert len(results) == 1
    assert results[0]["count"] == 1


def test_impossible_values_percentage_out_of_range():
    df = pd.DataFrame({"completion_rate": [0.5, 110.0, -5.0]})
    results = impossible_values(df)
    assert results[0]["count"] == 2


def test_impossible_values_non_matching_column_skipped():
    df = pd.DataFrame({"city_code": [-1, -2, -3]})
    assert impossible_values(df) == []


def test_impossible_values_non_numeric_gracefully_skipped():
    df = pd.DataFrame({"age": ["old", "young", "ancient"]})
    assert impossible_values(df) == []
