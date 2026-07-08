"""Completeness checks — missing values, empty rows, null concentration."""
from __future__ import annotations

import pandas as pd

from engine.checks.base import (
    Category, Severity, CheckResult,
    build_result, _sample, _is_text,
)

_CAT = Category.COMPLETENESS
_NULL_CONC_THRESHOLD = 0.50


def missing_values(df: pd.DataFrame) -> list[CheckResult]:
    """Null and whitespace-only values per column."""
    results: list[CheckResult] = []
    total = len(df)
    for col in df.columns:
        series = df[col]
        mask = series.isna()
        if _is_text(series):
            non_null = series.dropna()
            if not non_null.empty:
                ws_idx = non_null[non_null.astype(str).str.strip() == ""].index
                if len(ws_idx):
                    mask = mask.copy()
                    mask.loc[ws_idx] = True
        count = int(mask.sum())
        if count == 0:
            continue
        pct = round(count / total * 100, 1)
        results.append(build_result(
            category    = _CAT,
            check       = "missing_values",
            column      = str(col),
            count       = count,
            total       = total,
            severity    = Severity.WARN,
            description = (
                f"{count:,} blank or null values in '{col}' "
                f"({pct}% of rows)."
            ),
        ))
    return results


def empty_rows(df: pd.DataFrame) -> list[CheckResult]:
    """Rows where every value is null or whitespace-only."""
    total = len(df)
    if total == 0:
        return []
    is_null = df.isna().copy()
    for col in df.columns:
        if not _is_text(df[col]):
            continue
        is_null[col] |= df[col].astype(str).str.strip().eq("").fillna(False)
    mask = is_null.all(axis=1)
    count = int(mask.sum())
    if count == 0:
        return []
    idx_samples = [f"row_index={i}" for i in mask[mask].index[:5].tolist()]
    return [build_result(
        category      = _CAT,
        check         = "empty_rows",
        column        = "__row__",
        count         = count,
        total         = total,
        severity      = Severity.BAD,
        description   = f"{count:,} entirely empty rows.",
        sample_values = idx_samples,
    )]


def null_concentration(
    df: pd.DataFrame,
    threshold: float = _NULL_CONC_THRESHOLD,
) -> list[CheckResult]:
    """Flag columns where the null rate exceeds `threshold`."""
    results: list[CheckResult] = []
    total = len(df)
    if total == 0:
        return []
    for col in df.columns:
        null_rate = df[col].isna().mean()
        if null_rate < threshold:
            continue
        count = int(df[col].isna().sum())
        pct = round(null_rate * 100, 1)
        results.append(build_result(
            category    = _CAT,
            check       = "null_concentration",
            column      = str(col),
            count       = count,
            total       = total,
            severity    = Severity.BAD,
            description = (
                f"'{col}' is {pct}% null — exceeds the "
                f"{threshold * 100:.0f}% concentration threshold."
            ),
            threshold   = str(threshold),
        ))
    return results
