"""Tests for UI component helpers."""

from ui.components import quality_score


def test_quality_score_uses_finding_total_when_available():
    results = [
        {
            "check": "schema_missing_column",
            "severity": "bad",
            "count": 1,
            "total": 1,
        }
    ]

    assert quality_score(results, total_rows=1_000) == 0


def test_quality_score_falls_back_to_dataset_rows_without_finding_total():
    results = [
        {
            "check": "missing_values",
            "severity": "warn",
            "count": 10,
        }
    ]

    assert quality_score(results, total_rows=100) == 94


def test_quality_score_skips_zero_finding_total():
    results = [
        {
            "check": "missing_values",
            "severity": "warn",
            "count": 10,
            "total": 0,
        }
    ]

    assert quality_score(results, total_rows=100) == 94
