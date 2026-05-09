"""Format / pattern checks — inconsistencies, casing, whitespace."""
from __future__ import annotations

import re

import pandas as pd

from engine.checks.base import (
    Category, Severity, CheckResult,
    build_result, _sample, _parse_dt_silent,
)

_CAT = Category.FORMAT

_DATE_HINT = re.compile(
    r"(date|dob|birth|created|updated|submitted|registered|opened|closed|issued|maturity)",
    re.IGNORECASE,
)


def format_inconsistencies(df: pd.DataFrame) -> list[CheckResult]:
    """Detect column-name-as-value, numeric-as-text, and mixed-date patterns.

    Emits three distinct check names so the summary table and narrative can
    reference each issue type cleanly (preserved from original implementation).
    """
    results: list[CheckResult] = []
    total = len(df)

    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        str_vals = non_null.astype(str).str.strip()
        col_lower = str(col).strip().lower()

        # 1. Column name appears as a value
        name_match_mask = str_vals.str.lower() == col_lower
        name_count = int(name_match_mask.sum())
        if name_count:
            results.append(build_result(
                category      = _CAT,
                check         = "format_inconsistency_name_as_value",
                column        = str(col),
                count         = name_count,
                total         = total,
                severity      = Severity.INFO,
                description   = (
                    f"'{col}' has {name_count:,} cell(s) containing the "
                    "column name itself as a data value."
                ),
                sample_values = _sample(str_vals[name_match_mask]),
            ))

        # 2. Numeric values stored in an otherwise text column
        numeric_coerced  = pd.to_numeric(str_vals, errors="coerce")
        numeric_like     = numeric_coerced.notna()
        non_numeric_mask = ~numeric_like
        if numeric_like.any() and non_numeric_mask.any():
            num_count = int(numeric_like.sum())
            results.append(build_result(
                category      = _CAT,
                check         = "format_inconsistency_numeric_as_text",
                column        = str(col),
                count         = num_count,
                total         = total,
                severity      = Severity.WARN,
                description   = (
                    f"'{col}' is a text column but {num_count:,} value(s) "
                    "look like numeric values."
                ),
                sample_values = _sample(str_vals[numeric_like]),
            ))

        # 3. Mixed date / non-date strings
        if non_numeric_mask.any():
            non_numeric_vals = str_vals[non_numeric_mask]
            dt_parsed  = _parse_dt_silent(non_numeric_vals)
            date_mask  = dt_parsed.notna()
            non_date   = ~date_mask
            if date_mask.any() and non_date.any():
                d_count = int(date_mask.sum())
                results.append(build_result(
                    category      = _CAT,
                    check         = "format_inconsistency_date_mixed",
                    column        = str(col),
                    count         = d_count,
                    total         = total,
                    severity      = Severity.WARN,
                    description   = (
                        f"'{col}' contains {d_count:,} date-like value(s) "
                        "mixed with non-date strings."
                    ),
                    sample_values = _sample(non_numeric_vals[date_mask]),
                ))

    return results


def casing_inconsistency(df: pd.DataFrame) -> list[CheckResult]:
    """Flag string columns containing a mix of lower, upper, and title-case values.

    Only fires when two or more case styles are present AND the dominant style
    accounts for less than 95% of non-null values.
    """
    results: list[CheckResult] = []
    total = len(df)

    for col in df.select_dtypes(include="object").columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        n = len(series)
        lower_n = int(series.str.islower().sum())
        upper_n = int(series.str.isupper().sum())
        title_n = int(series.str.istitle().sum())

        present = [v for v in (lower_n, upper_n, title_n) if v > 0]
        if len(present) < 2:
            continue
        majority = max(present)
        if majority / n >= 0.95:
            continue

        minority_count = n - majority
        if lower_n == majority:
            minority_mask = ~series.str.islower()
        elif upper_n == majority:
            minority_mask = ~series.str.isupper()
        else:
            minority_mask = ~series.str.istitle()

        results.append(build_result(
            category      = _CAT,
            check         = "casing_inconsistency",
            column        = str(col),
            count         = minority_count,
            total         = total,
            severity      = Severity.WARN,
            description   = (
                f"'{col}' has inconsistent casing "
                f"(lower: {lower_n}, upper: {upper_n}, title: {title_n})."
            ),
            sample_values = _sample(df[col].dropna()[minority_mask]),
        ))
    return results


def whitespace_issues(df: pd.DataFrame) -> list[CheckResult]:
    """Detect leading or trailing whitespace in string columns."""
    results: list[CheckResult] = []
    total = len(df)

    for col in df.select_dtypes(include="object").columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        mask = series != series.str.strip()
        count = int(mask.sum())
        if count == 0:
            continue
        results.append(build_result(
            category      = _CAT,
            check         = "whitespace_issues",
            column        = str(col),
            count         = count,
            total         = total,
            severity      = Severity.INFO,
            description   = (
                f"{count:,} values in '{col}' have leading or "
                "trailing whitespace."
            ),
            sample_values = _sample(series[mask]),
        ))
    return results
