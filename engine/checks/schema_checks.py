"""Schema-driven checks — migrated from engine/schema.py + unexpected-columns check."""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

import pandas as pd

from engine.checks.base import (
    Category, Severity, CheckResult,
    build_result, _sample, _parse_dt_silent,
)

logger = logging.getLogger(__name__)

_CAT = Category.SCHEMA
_VALID_TYPES = {"text", "numeric", "date", "identifier"}


def run_schema_checks(
    schema_dict: dict[str, dict[str, Any]],
    df: pd.DataFrame,
) -> list[CheckResult]:
    """Entry point: run all schema-driven checks against `df`."""
    if not schema_dict:
        return []

    results: list[CheckResult] = []
    total = len(df)

    # Unexpected columns (in df but not in schema)
    results.extend(_unexpected_columns(schema_dict, df, total))

    for col_name, spec in schema_dict.items():
        if col_name not in df.columns:
            results.append(build_result(
                category      = _CAT,
                check         = "schema_missing_column",
                column        = col_name,
                count         = 1,
                total         = 1,
                severity      = Severity.BAD,
                description   = (
                    f"Required column '{col_name}' is absent from the dataset."
                ),
                sample_values = [],
                notes         = spec.get("notes"),
            ))
            continue

        series       = df[col_name]
        non_null     = series.dropna()
        expected_type = spec.get("expected_type")

        if expected_type:
            results.extend(_check_type(non_null, col_name, expected_type, total, spec))

        if spec.get("max_value") is not None or spec.get("min_value") is not None:
            results.extend(_check_range(non_null, col_name, expected_type, spec, total))

        if spec.get("allowed_formats"):
            results.extend(
                _check_format(non_null, col_name, spec["allowed_formats"], total, spec)
            )

    return results


# ── Internal helpers ──────────────────────────────────────────

def _unexpected_columns(
    schema_dict: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    total: int,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    schema_cols = set(schema_dict.keys())
    extra = [str(c) for c in df.columns if str(c) not in schema_cols]
    for col in extra:
        results.append(build_result(
            category    = _CAT,
            check       = "schema_unexpected_columns",
            column      = col,
            count       = 1,
            total       = 1,
            severity    = Severity.INFO,
            description = (
                f"Column '{col}' is present in the dataset but not "
                "declared in the schema."
            ),
        ))
    return results


def _check_type(
    non_null: pd.Series,
    col: str,
    expected_type: str,
    total: int,
    spec: dict[str, Any],
) -> list[CheckResult]:
    if non_null.empty or expected_type not in _VALID_TYPES:
        return []

    if expected_type == "numeric":
        coerced  = pd.to_numeric(non_null, errors="coerce")
        mismatch = coerced.isna()
    elif expected_type == "date":
        coerced  = _parse_dt_silent(non_null)
        mismatch = coerced.isna()
    elif expected_type == "text":
        numeric_coerced = pd.to_numeric(
            non_null.astype(str).str.strip(), errors="coerce"
        )
        mismatch = numeric_coerced.notna()
    elif expected_type == "identifier":
        mismatch = non_null.astype(str).str.strip().eq("")
    else:
        return []

    count = int(mismatch.sum())
    if count == 0:
        return []
    return [build_result(
        category      = _CAT,
        check         = f"schema_type_mismatch_{expected_type}",
        column        = col,
        count         = count,
        total         = total,
        severity      = Severity.WARN,
        description   = (
            f"'{col}' has {count:,} value(s) that do not match the "
            f"expected type '{expected_type}'."
        ),
        sample_values = _sample(non_null[mismatch]),
        notes         = spec.get("notes"),
    )]


def _check_range(
    non_null: pd.Series,
    col: str,
    expected_type: Optional[str],
    spec: dict[str, Any],
    total: int,
) -> list[CheckResult]:
    if non_null.empty:
        return []

    results: list[CheckResult] = []
    is_date = expected_type == "date"

    if is_date:
        values = _parse_dt_silent(non_null)
        min_v  = (pd.to_datetime(spec["min_value"], errors="coerce")
                  if spec.get("min_value") else None)
        max_v  = (pd.to_datetime(spec["max_value"], errors="coerce")
                  if spec.get("max_value") else None)
    else:
        values = pd.to_numeric(non_null, errors="coerce")
        min_v  = _to_float(spec.get("min_value"))
        max_v  = _to_float(spec.get("max_value"))

    valid = values.notna()

    def _is_null_ts(v: Any) -> bool:
        return isinstance(v, pd.Timestamp) and pd.isna(v)

    if min_v is not None and not _is_null_ts(min_v):
        below = valid & (values < min_v)
        count = int(below.sum())
        if count:
            results.append(build_result(
                category      = _CAT,
                check         = "schema_below_min",
                column        = col,
                count         = count,
                total         = total,
                severity      = Severity.WARN,
                description   = (
                    f"'{col}' has {count:,} value(s) below the minimum "
                    f"threshold of {min_v}."
                ),
                sample_values = _sample(non_null[below]),
                notes         = spec.get("notes"),
                threshold     = str(min_v),
            ))

    if max_v is not None and not _is_null_ts(max_v):
        above = valid & (values > max_v)
        count = int(above.sum())
        if count:
            results.append(build_result(
                category      = _CAT,
                check         = "schema_above_max",
                column        = col,
                count         = count,
                total         = total,
                severity      = Severity.WARN,
                description   = (
                    f"'{col}' has {count:,} value(s) above the maximum "
                    f"threshold of {max_v}."
                ),
                sample_values = _sample(non_null[above]),
                notes         = spec.get("notes"),
                threshold     = str(max_v),
            ))

    return results


def _check_format(
    non_null: pd.Series,
    col: str,
    pattern: str,
    total: int,
    spec: dict[str, Any],
) -> list[CheckResult]:
    if non_null.empty:
        return []
    try:
        re.compile(pattern)
    except re.error as exc:
        logger.warning("Invalid regex '%s' for column '%s': %s", pattern, col, exc)
        return []

    str_vals = non_null.astype(str).str.strip()
    mismatch  = ~str_vals.str.fullmatch(pattern, na=False)
    count     = int(mismatch.sum())
    if count == 0:
        return []
    return [build_result(
        category      = _CAT,
        check         = "schema_format_violation",
        column        = col,
        count         = count,
        total         = total,
        severity      = Severity.WARN,
        description   = (
            f"'{col}' has {count:,} value(s) that do not match the "
            f"required format pattern."
        ),
        sample_values = _sample(non_null[mismatch]),
        notes         = spec.get("notes"),
        pattern       = pattern,
    )]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
