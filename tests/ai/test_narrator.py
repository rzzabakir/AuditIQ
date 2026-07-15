from ai import narrator


def test_generate_narrative_uses_deterministic_fallback_without_api_key(monkeypatch):
    monkeypatch.setattr(narrator, "_load_api_key", lambda: None)

    narrative = narrator.generate_narrative(
        [
            {
                "check": "missing_values",
                "column": "Age",
                "count": 3,
                "total": 10,
                "sample_values": [],
            }
        ],
        {"filename": "sample.csv", "rows": 10, "columns": 1},
    )

    assert "Age:" in narrative
    assert "3 records (30.0% of rows) showed missing values" in narrative


def test_generate_narrative_labels_multi_column_findings(monkeypatch):
    monkeypatch.setattr(narrator, "_load_api_key", lambda: None)

    narrative = narrator.generate_narrative(
        [
            {
                "check": "duplicate_column_combos",
                "column": "__multi__",
                "columns": ["customer_id", "customer_name"],
                "count": 4,
                "total": 20,
                "sample_values": [],
            }
        ],
        {"filename": "sample.csv", "rows": 20, "columns": 2},
    )

    assert "__multi__" not in narrative
    assert "customer_id + customer_name:" in narrative
