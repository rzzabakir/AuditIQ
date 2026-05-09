"""Shared primitives for all check families.

Import pattern inside check modules:
    from engine.checks.base import (
        Category, Severity, CheckResult,
        build_result, _sample, _parse_dt_silent,
    )
"""
from __future__ import annotations

import warnings
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict

import pandas as pd


# ── Taxonomy ─────────────────────────────────────────────────

class Category(str, Enum):
    COMPLETENESS  = "completeness"
    UNIQUENESS    = "uniqueness"
    VALIDITY      = "validity"
    RANGE         = "range"
    FORMAT        = "format"
    CONSISTENCY   = "consistency"
    REFERENTIAL   = "referential"
    BUSINESS_RULE = "business_rule"
    TIMELINESS    = "timeliness"
    DISTRIBUTION  = "distribution"
    CROSS_FIELD   = "cross_field"
    SCHEMA        = "schema"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    BAD  = "bad"


# ── Result contract ───────────────────────────────────────────

class CheckResult(TypedDict, total=False):
    # Identity
    check_id:      str        # "{category}.{check}"  e.g. "completeness.missing_values"
    category:      str        # Category enum value
    check:         str        # short name — preserved for backward compat
    # Scope
    column:        str        # column name, "__row__", or "__multi__"
    columns:       list[str]  # populated for multi-column checks
    # Signal
    severity:      str        # Severity enum value
    count:         int
    total:         int
    pct:           float      # count/total * 100, rounded to 2 dp
    description:   str        # one human-readable sentence
    sample_values: list
    # Domain extras (backward compat — optional, populated only where relevant)
    notes:         str
    threshold:     str
    pattern:       str
    type_counts:   dict
    majority_type: str


MAX_SAMPLES = 5


# ── Builder ───────────────────────────────────────────────────

def build_result(
    category: str,
    check: str,
    column: str,
    count: int,
    total: int,
    severity: str,
    description: str,
    sample_values: list | None = None,
    **extras: Any,
) -> CheckResult:
    """Construct a fully-populated CheckResult from required fields.

    All new fields (check_id, category, severity, pct, description) are
    additive — the existing keys (check, column, count, total, sample_values)
    are always present for backward compatibility.

    Enum instances are normalised to their plain string `.value` so that
    downstream consumers always see plain strings regardless of how the
    caller passed the arguments.
    """
    cat_str = category.value if hasattr(category, "value") else str(category)
    sev_str = severity.value if hasattr(severity, "value") else str(severity)
    return CheckResult(
        check_id      = f"{cat_str}.{check}",
        category      = cat_str,
        check         = check,
        column        = column,
        severity      = sev_str,
        count         = count,
        total         = total,
        pct           = round(count / total * 100, 2) if total else 0.0,
        description   = description,
        sample_values = sample_values or [],
        **extras,
    )


# ── Shared utilities ──────────────────────────────────────────

def _to_native(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    return value


def _sample(series: pd.Series, limit: int = MAX_SAMPLES) -> list:
    """Return up to `limit` unique, JSON-safe sample values."""
    if series is None or len(series) == 0:
        return []
    cleaned = series.dropna()
    if cleaned.empty:
        return []
    return [_to_native(v) for v in cleaned.unique().tolist()[:limit]]


def _parse_dt_silent(series: pd.Series) -> pd.Series:
    """pd.to_datetime with warnings suppressed and mixed-format tolerance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return pd.to_datetime(series, errors="coerce")
