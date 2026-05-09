"""Shared pytest fixtures for engine check tests."""
import pandas as pd
import pytest


@pytest.fixture
def clean_df():
    """A well-formed DataFrame with no data quality issues."""
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "customer_name": ["Alice", "Bob", "Carol", "David", "Eve"],
        "email": [
            "alice@example.com",
            "bob@example.com",
            "carol@example.com",
            "david@example.com",
            "eve@example.com",
        ],
        "age": [28, 34, 45, 52, 31],
        "purchase_amount": [120.0, 89.5, 340.0, 210.75, 55.0],
        "signup_date": ["2023-01-10", "2023-02-14", "2023-03-22", "2023-04-05", "2023-05-18"],
        "status": ["active", "active", "inactive", "active", "inactive"],
    })


@pytest.fixture
def nulls_df():
    """A DataFrame with null values in multiple columns."""
    return pd.DataFrame({
        "customer_id": ["C001", "C002", None, "C004", None],
        "customer_name": ["Alice", None, "Carol", None, "Eve"],
        "email": [None, "bob@example.com", None, "david@example.com", None],
        "age": [28.0, None, 45.0, None, 31.0],
    })


@pytest.fixture
def duplicate_rows_df():
    """A DataFrame containing exact full-row duplicates."""
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C001"],
        "customer_name": ["Alice", "Bob", "Alice"],
        "age": [28, 34, 28],
    })


@pytest.fixture
def empty_df():
    """An empty DataFrame with column headers but no rows."""
    return pd.DataFrame({
        "customer_id": pd.Series([], dtype=str),
        "age": pd.Series([], dtype=float),
    })
