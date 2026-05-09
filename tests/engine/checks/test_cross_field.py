"""Tests for engine.checks.cross_field."""
import pandas as pd
import pytest

from engine.checks.cross_field import start_after_end


def test_no_start_or_end_columns():
    df = pd.DataFrame({"score": [1, 2], "city": ["A", "B"]})
    assert start_after_end(df) == []


def test_valid_date_pairs():
    df = pd.DataFrame({
        "start_date": ["2023-01-01", "2023-03-15"],
        "end_date":   ["2023-06-30", "2023-12-31"],
    })
    assert start_after_end(df) == []


def test_detects_start_after_end():
    df = pd.DataFrame({
        "start_date": ["2023-01-01", "2023-12-31"],
        "end_date":   ["2023-06-30", "2023-01-01"],
    })
    results = start_after_end(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "start_after_end"
    assert r["count"] == 1
    assert r["severity"] == "bad"


def test_all_rows_violate():
    df = pd.DataFrame({
        "open_date":  ["2023-12-31", "2023-11-30"],
        "close_date": ["2023-01-01", "2023-01-01"],
    })
    results = start_after_end(df)
    assert results[0]["count"] == 2


def test_equal_dates_not_flagged():
    df = pd.DataFrame({
        "start_date": ["2023-06-01"],
        "end_date":   ["2023-06-01"],
    })
    assert start_after_end(df) == []


def test_nulls_in_date_columns_skipped():
    df = pd.DataFrame({
        "start_date": [None, "2023-01-01"],
        "end_date":   [None, "2023-06-30"],
    })
    assert start_after_end(df) == []


def test_empty_dataframe():
    df = pd.DataFrame({"start_date": [], "end_date": []})
    assert start_after_end(df) == []


def test_columns_reported_in_result():
    df = pd.DataFrame({
        "issue_date":     ["2023-12-31"],
        "maturity_date":  ["2023-01-01"],
    })
    results = start_after_end(df)
    assert len(results) == 1
    r = results[0]
    assert "issue_date" in r["columns"]
    assert "maturity_date" in r["columns"]
