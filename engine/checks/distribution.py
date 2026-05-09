"""Distribution / anomaly checks — outliers, value concentration."""
from __future__ import annotations

import pandas as pd

from engine.checks.base import (
    Category, Severity, CheckResult,
    build_result, _sample,
)

_CAT = Category.DISTRIBUTION

_IQR_FACTOR         = 1.5    # standard Tukey fence
_MIN_VALUES_NEEDED  = 10     # skip columns with too few data points
_CONCENTRATION_THRESHOLD = 0.90  # flag if one value accounts for ≥90%


def outliers(df: pd.DataFrame) -> list[CheckResult]:
    """Flag numeric columns with values outside the 1.5 × IQR Tukey fences.

    Skips columns with fewer than MIN_VALUES_NEEDED non-null values to avoid
    noisy results on sparse or near-empty columns.
    """
    results: list[CheckResult] = []
    total = len(df)

    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < _MIN_VALUES_NEEDED:
            continue

        q1, q3  = series.quantile([0.25, 0.75])
        iqr     = q3 - q1
        if iqr == 0:
            continue

        fence_lo = q1 - _IQR_FACTOR * iqr
        fence_hi = q3 + _IQR_FACTOR * iqr

        # Use the original (non-dropna) series so mask aligns with df index
        numeric = pd.to_numeric(df[col], errors="coerce")
        out_mask = numeric.notna() & (
            (numeric < fence_lo) | (numeric > fence_hi)
        )
        count = int(out_mask.sum())
        if count == 0:
            continue

        results.append(build_result(
            category      = _CAT,
            check         = "outliers",
            column        = str(col),
            count         = count,
            total         = total,
            severity      = Severity.WARN,
            description   = (
                f"'{col}' has {count:,} statistical outlier(s) outside the "
                f"Tukey fences [{fence_lo:.2f}, {fence_hi:.2f}]."
            ),
            sample_values = _sample(df.loc[out_mask, col]),
            notes         = f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}",
        ))

    return results


def value_concentration(
    df: pd.DataFrame,
    threshold: float = _CONCENTRATION_THRESHOLD,
) -> list[CheckResult]:
    """Flag columns where a single value dominates ≥ threshold of non-null rows.

    Applies to all column types. High concentration may indicate a defaulted
    or corrupt field that should be reviewed.
    """
    results: list[CheckResult] = []
    total = len(df)
    if total == 0:
        return []

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        top_count = int(series.value_counts().iloc[0])
        top_rate  = top_count / len(series)
        if top_rate < threshold:
            continue
        top_value = series.value_counts().index[0]
        results.append(build_result(
            category      = _CAT,
            check         = "value_concentration",
            column        = str(col),
            count         = top_count,
            total         = total,
            severity      = Severity.INFO,
            description   = (
                f"'{col}' has one value ({top_value!r}) accounting for "
                f"{round(top_rate * 100, 1)}% of non-null entries — "
                "possible default or corrupt data."
            ),
            sample_values = [str(top_value)],
            threshold     = str(threshold),
        ))

    return results
