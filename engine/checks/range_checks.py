"""Range / boundary checks — future dates, impossible values."""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from engine.checks.base import (
    Category, Severity, CheckResult,
    build_result, _sample, _parse_dt_silent,
)

_CAT = Category.RANGE

_DATE_KEYWORDS = re.compile(
    r"(date|dob|birth|created|updated|submitted|registered|opened|closed|issued|maturity)",
    re.IGNORECASE,
)

# Impossible-value rules: (column keyword regex, rule description, test function)
# test_fn(series: pd.Series) -> pd.Series[bool]  (True = bad)
_IMPOSSIBLE_RULES: list[tuple[re.Pattern, str, str, object]] = [
    (
        re.compile(r"\bage\b", re.IGNORECASE),
        "age",
        "age ≤ 0 or > 120",
        lambda s: ~s.between(1, 120),
    ),
    (
        re.compile(
            r"(amount|balance|price|cost|salary|income|fee|charge|payment)",
            re.IGNORECASE,
        ),
        "amount/balance",
        "negative value where negative is invalid",
        lambda s: s < 0,
    ),
    (
        re.compile(r"(pct|percent|rate|ratio|proportion)", re.IGNORECASE),
        "percentage/rate",
        "value outside 0–100 range for a percentage/rate column",
        lambda s: ~s.between(0, 100),
    ),
]


def future_dates(df: pd.DataFrame) -> list[CheckResult]:
    """Values after today's date in columns whose name implies a date."""
    results: list[CheckResult] = []
    total = len(df)
    today = pd.Timestamp(datetime.now().date())

    for col in df.columns:
        if not _DATE_KEYWORDS.search(str(col)):
            continue
        series = df[col]
        parsed = _parse_dt_silent(series)
        future_mask = parsed.notna() & (parsed > today)
        count = int(future_mask.sum())
        if count == 0:
            continue
        results.append(build_result(
            category      = _CAT,
            check         = "future_dates",
            column        = str(col),
            count         = count,
            total         = total,
            severity      = Severity.WARN,
            description   = (
                f"'{col}' has {count:,} value(s) with a date after today "
                f"({today.date()})."
            ),
            sample_values = _sample(parsed[future_mask]),
        ))
    return results


def impossible_values(df: pd.DataFrame) -> list[CheckResult]:
    """Flag numeric values that violate domain-specific boundary rules.

    Applies keyword-based heuristics:
    - age columns: must be 1–120
    - amount/balance/price/cost/salary/fee columns: must be ≥ 0
    - pct/rate/ratio columns: must be 0–100
    """
    results: list[CheckResult] = []
    total = len(df)

    for col in df.columns:
        col_str = str(col)
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            continue

        for pattern, _label, rule_desc, test_fn in _IMPOSSIBLE_RULES:
            if not pattern.search(col_str):
                continue
            bad_mask_valid = test_fn(valid)
            count = int(bad_mask_valid.sum())
            if count == 0:
                continue
            results.append(build_result(
                category      = _CAT,
                check         = "impossible_values",
                column        = col_str,
                count         = count,
                total         = total,
                severity      = Severity.BAD,
                description   = (
                    f"'{col_str}' has {count:,} value(s) that violate a "
                    f"domain rule: {rule_desc}."
                ),
                sample_values = _sample(valid[bad_mask_valid]),
                notes         = rule_desc,
            ))
            break  # one rule per column

    return results
