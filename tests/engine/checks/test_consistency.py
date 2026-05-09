"""Tests for engine.checks.consistency."""
import pandas as pd
import pytest

from engine.checks.consistency import id_to_name_conflicts


def test_no_id_or_name_columns():
    df = pd.DataFrame({"score": [1, 2, 3], "city": ["A", "B", "C"]})
    assert id_to_name_conflicts(df) == []


def test_consistent_id_to_name_mapping():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C1", "C2"],
        "customer_name": ["Alice", "Bob", "Alice", "Bob"],
    })
    assert id_to_name_conflicts(df) == []


def test_detects_id_mapping_to_multiple_names():
    df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "customer_name": ["Alice", "Alice Smith", "Bob"],
    })
    results = id_to_name_conflicts(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "id_to_name_conflicts"
    assert r["count"] == 1  # one conflicting ID value (C1)
    assert r["severity"] == "warn"


def test_multiple_conflicts_counted():
    df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2", "C2"],
        "customer_name": ["Alice", "Alice X", "Bob", "Robert"],
    })
    results = id_to_name_conflicts(df)
    assert results[0]["count"] == 2


def test_empty_dataframe():
    df = pd.DataFrame({"customer_id": [], "customer_name": []})
    assert id_to_name_conflicts(df) == []


def test_all_nulls_skipped():
    df = pd.DataFrame({
        "customer_id": [None, None],
        "customer_name": [None, None],
    })
    assert id_to_name_conflicts(df) == []


def test_column_name_reported():
    df = pd.DataFrame({
        "record_id": ["R1", "R1"],
        "record_name": ["Foo", "Bar"],
    })
    results = id_to_name_conflicts(df)
    assert results[0]["column"] == "record_id"
