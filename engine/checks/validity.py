"""Validity / type correctness checks — mixed types, invalid dates, malformed IDs."""
from __future__ import annotations

import re

import pandas as pd

from engine.checks.base import (
    Category, Severity, CheckResult,
    build_result, _sample, _parse_dt_silent, _is_text,
)

_CAT = Category.VALIDITY

_MIXED_TYPE_THRESHOLD = 0.05

# Date-like column name keywords
_DATE_KEYWORDS = re.compile(
    r"(date|dob|birth|created|updated|submitted|registered|opened|closed|issued|maturity)",
    re.IGNORECASE,
)
# Fraction of values that must parse successfully for the column to be "valid"
_DATE_PARSE_MIN_RATE = 0.90

# Email / phone column keywords
_EMAIL_KEYWORDS = re.compile(r"(email|e[-_]?mail)", re.IGNORECASE)
_PHONE_KEYWORDS = re.compile(r"(phone|tel|mobile|cell|contact)", re.IGNORECASE)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def mixed_types(df: pd.DataFrame) -> list[CheckResult]:
    """Text columns whose minority inferred type exceeds 5% of non-nulls."""
    results: list[CheckResult] = []
    total = len(df)
    int_pat   = r"^-?\d+$"
    float_pat = r"^-?\d+\.\d+$"
    sci_pat   = r"^-?\d+(\.\d+)?[eE][-+]?\d+$"

    for col in df.columns:
        series = df[col]
        if not _is_text(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        str_vals = non_null.astype(str).str.strip()

        int_mask   = str_vals.str.match(int_pat, na=False)
        float_mask = (str_vals.str.match(float_pat, na=False)
                      | str_vals.str.match(sci_pat, na=False))
        numeric_mask = int_mask | float_mask

        non_numeric = str_vals[~numeric_mask]
        datetime_mask = pd.Series(False, index=non_null.index)
        if not non_numeric.empty:
            dt_parsed = _parse_dt_silent(non_numeric)
            datetime_mask.loc[non_numeric.index] = dt_parsed.notna()

        str_mask = ~numeric_mask & ~datetime_mask

        counts = {
            "int":      int(int_mask.sum()),
            "float":    int(float_mask.sum()),
            "datetime": int(datetime_mask.sum()),
            "string":   int(str_mask.sum()),
        }
        present = {k: v for k, v in counts.items() if v > 0}
        if len(present) < 2:
            continue
        total_nn = sum(counts.values())
        if total_nn == 0:
            continue
        majority_type = max(present, key=present.get)
        minority_count = total_nn - counts[majority_type]
        if minority_count / total_nn < _MIXED_TYPE_THRESHOLD:
            continue

        minority_mask = pd.Series(False, index=non_null.index)
        for t, m in (
            ("int", int_mask),
            ("float", float_mask),
            ("datetime", datetime_mask),
            ("string", str_mask),
        ):
            if t != majority_type:
                minority_mask = minority_mask | m

        results.append(build_result(
            category      = _CAT,
            check         = "mixed_types",
            column        = str(col),
            count         = minority_count,
            total         = total,
            severity      = Severity.WARN,
            description   = (
                f"'{col}' contains mixed data types: majority is "
                f"{majority_type} but {minority_count:,} value(s) "
                "differ."
            ),
            sample_values = _sample(non_null[minority_mask]),
            type_counts   = counts,
            majority_type = majority_type,
        ))
    return results


def invalid_dates(df: pd.DataFrame) -> list[CheckResult]:
    """Flag date-keyword columns where a significant fraction of values do not parse.

    Fires only when the column is object/string typed (not already a datetime dtype)
    AND fewer than DATE_PARSE_MIN_RATE of non-null values parse successfully.
    """
    results: list[CheckResult] = []
    total = len(df)
    for col in df.columns:
        if not _DATE_KEYWORDS.search(str(col)):
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        parsed = _parse_dt_silent(non_null.astype(str))
        parse_rate = parsed.notna().mean()
        if parse_rate >= _DATE_PARSE_MIN_RATE:
            continue
        # Only report the unparseable fraction
        bad_mask = parsed.isna()
        count = int(bad_mask.sum())
        results.append(build_result(
            category      = _CAT,
            check         = "invalid_dates",
            column        = str(col),
            count         = count,
            total         = total,
            severity      = Severity.WARN,
            description   = (
                f"'{col}' has {count:,} value(s) that cannot be parsed as a "
                f"date ({round((1 - parse_rate) * 100, 1)}% invalid)."
            ),
            sample_values = _sample(non_null[bad_mask.values]),
        ))
    return results


def malformed_identifiers(df: pd.DataFrame) -> list[CheckResult]:
    """Check email and phone columns against simple structural patterns.

    Email rule: must match `local@domain.tld` (no spaces, one @).
    Phone rule: after stripping non-digit characters, the digit count must
    be between 7 and 15 (ITU-T E.164 range).
    """
    results: list[CheckResult] = []
    total = len(df)

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        if not _is_text(series):
            continue
        non_null = series.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        if non_null.empty:
            continue

        if _EMAIL_KEYWORDS.search(col_str):
            bad_mask = ~non_null.str.match(_EMAIL_RE.pattern, na=False)
            count = int(bad_mask.sum())
            if count:
                results.append(build_result(
                    category      = _CAT,
                    check         = "malformed_identifiers",
                    column        = col_str,
                    count         = count,
                    total         = total,
                    severity      = Severity.WARN,
                    description   = (
                        f"'{col_str}' has {count:,} value(s) that do not "
                        "match a valid email address format."
                    ),
                    sample_values = _sample(non_null[bad_mask]),
                    notes         = "email: expected local@domain.tld",
                ))

        elif _PHONE_KEYWORDS.search(col_str):
            digit_counts = non_null.str.replace(r"\D", "", regex=True).str.len()
            bad_mask = ~digit_counts.between(7, 15)
            count = int(bad_mask.sum())
            if count:
                results.append(build_result(
                    category      = _CAT,
                    check         = "malformed_identifiers",
                    column        = col_str,
                    count         = count,
                    total         = total,
                    severity      = Severity.WARN,
                    description   = (
                        f"'{col_str}' has {count:,} value(s) with an "
                        "unexpected digit count (expected 7–15 digits)."
                    ),
                    sample_values = _sample(non_null[bad_mask]),
                    notes         = "phone: expected 7–15 digits after stripping separators",
                ))

    return results
