"""Schema / data dictionary loader.

Validation logic (match_schema_to_df) has moved to engine/checks/schema_checks.py.
This module is now a pure loader.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".xlsx", ".csv"}
REQUIRED_COLUMNS = ("column_name",)
OPTIONAL_COLUMNS = ("expected_type", "max_value", "min_value", "allowed_formats", "notes")
VALID_TYPES = {"text", "numeric", "date", "identifier"}


def load_schema(filepath: str | Path) -> dict[str, dict[str, Any]]:
    """Load a schema file (.xlsx or .csv) and return a dict keyed by column_name."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported schema file type '{ext}'. "
            f"Expected: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".xlsx":
        raw = pd.read_excel(path, engine="openpyxl")
    else:
        try:
            raw = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="latin-1")

    raw.columns = [str(c).strip().lower() for c in raw.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Schema file missing required column(s): {missing}. "
            f"Found columns: {list(raw.columns)}"
        )

    schema: dict[str, dict[str, Any]] = {}
    for _, row in raw.iterrows():
        col_name = _clean(row.get("column_name"))
        if col_name is None:
            continue

        expected_type = _clean(row.get("expected_type"))
        if expected_type is not None:
            expected_type = expected_type.lower()
            if expected_type not in VALID_TYPES:
                logger.warning(
                    "Unknown expected_type '%s' for column '%s'; ignoring.",
                    expected_type, col_name,
                )
                expected_type = None

        schema[col_name] = {
            "expected_type":   expected_type,
            "max_value":       _clean(row.get("max_value")),
            "min_value":       _clean(row.get("min_value")),
            "allowed_formats": _clean(row.get("allowed_formats")),
            "notes":           _clean(row.get("notes")),
        }

    logger.info("Loaded schema with %d column specs from %s", len(schema), path.name)
    return schema


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s
