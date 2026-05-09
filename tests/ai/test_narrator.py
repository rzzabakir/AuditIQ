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
