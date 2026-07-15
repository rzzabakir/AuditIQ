from engine.checks.base import display_column


def test_display_column_passes_through_plain_names():
    assert display_column({"column": "Age"}) == "Age"


def test_display_column_resolves_row_sentinel():
    assert display_column({"column": "__row__"}) == "Dataset (all rows)"


def test_display_column_joins_multi_column_names():
    result = {"column": "__multi__", "columns": ["customer_id", "customer_name"]}
    assert display_column(result) == "customer_id + customer_name"


def test_display_column_multi_without_columns_falls_back():
    assert display_column({"column": "__multi__"}) == "Multiple columns"
