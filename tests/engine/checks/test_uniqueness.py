"""Tests for engine.checks.uniqueness."""
import pandas as pd
import pytest

from engine.checks.uniqueness import (
    duplicate_column_combos,
    duplicate_keys_inferred,
    duplicate_rows,
)


# ── duplicate_rows ────────────────────────────────────────────


def test_duplicate_rows_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert duplicate_rows(df) == []


def test_duplicate_rows_detects_exact_duplicates():
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    results = duplicate_rows(df)
    assert len(results) == 1
    r = results[0]
    assert r["count"] == 1
    assert r["total"] == 3
    assert r["severity"] == "bad"
    assert r["column"] == "__row__"


def test_duplicate_rows_multiple_dupes():
    df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "x"]})
    results = duplicate_rows(df)
    assert results[0]["count"] == 2


def test_duplicate_rows_empty_dataframe():
    df = pd.DataFrame({"a": []})
    assert duplicate_rows(df) == []


def test_duplicate_rows_sample_values_are_row_indices():
    df = pd.DataFrame({"a": [1, 1], "b": ["x", "x"]})
    r = duplicate_rows(df)[0]
    assert any("row_index" in str(v) for v in r["sample_values"])


# ── duplicate_keys_inferred ───────────────────────────────────


def test_duplicate_keys_inferred_all_unique():
    df = pd.DataFrame({"customer_id": ["C001", "C002", "C003"]})
    assert duplicate_keys_inferred(df) == []


def test_duplicate_keys_inferred_fires_on_duplicate_id():
    df = pd.DataFrame({"customer_id": ["C001", "C002", "C001", "C003"]})
    results = duplicate_keys_inferred(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "duplicate_keys_inferred"
    assert r["count"] == 1
    assert r["severity"] == "bad"


def test_duplicate_keys_inferred_skips_non_id_columns():
    df = pd.DataFrame({"status": ["active", "active", "active"]})
    assert duplicate_keys_inferred(df) == []


def test_duplicate_keys_inferred_skips_low_cardinality_columns():
    # Column named with id-keyword but only 2 unique values out of 10 — below 70% threshold
    df = pd.DataFrame({"branch_id": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]})
    assert duplicate_keys_inferred(df) == []


def test_duplicate_keys_inferred_ignores_nulls():
    df = pd.DataFrame({"record_id": ["R1", None, "R1", "R2", "R3"]})
    results = duplicate_keys_inferred(df)
    # R1 is duplicated — should fire; None values are ignored
    assert len(results) == 1
    assert results[0]["count"] == 1


# ── duplicate_column_combos ───────────────────────────────────


def test_duplicate_column_combos_no_pairs():
    df = pd.DataFrame({"score": [1, 2, 3]})
    assert duplicate_column_combos(df) == []


def test_duplicate_column_combos_no_duplicates():
    df = pd.DataFrame({"customer_id": ["C1", "C2"], "customer_name": ["Alice", "Bob"]})
    assert duplicate_column_combos(df) == []


def test_duplicate_column_combos_detects_verbatim_repeat():
    df = pd.DataFrame({
        "customer_id":   ["C1", "C1", "C2"],
        "customer_name": ["Alice", "Alice", "Bob"],
    })
    results = duplicate_column_combos(df)
    assert len(results) == 1
    r = results[0]
    assert r["check"] == "duplicate_column_combos"
    assert r["count"] == 1
    assert r["severity"] == "warn"


def test_duplicate_column_combos_empty_dataframe():
    df = pd.DataFrame({"customer_id": [], "customer_name": []})
    assert duplicate_column_combos(df) == []
