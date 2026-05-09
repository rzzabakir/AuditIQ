"""Cross-field logic checks — start date after end date."""
from __future__ import annotations

import re

import pandas as pd

from engine.checks.base import (
    Category, Severity, CheckResult,
    build_result, _sample, _parse_dt_silent,
)

_CAT = Category.CROSS_FIELD

_START_PATTERN = re.compile(
    r"(^|_)(start|open|issue|begin|from|created|inception)($|_)",
    re.IGNORECASE,
)
_END_PATTERN = re.compile(
    r"(^|_)(end|close|maturity|expiry|expire|to|terminated|closed)($|_)",
    re.IGNORECASE,
)


def start_after_end(df: pd.DataFrame) -> list[CheckResult]:
    """Flag rows where a start-like date column is later than an end-like date column.

    Heuristically pairs start and end columns by name. For each valid pair both
    columns must parse as dates. Fires once per (start_col, end_col) pair.
    """
    results: list[CheckResult] = []
    total = len(df)
    if total == 0:
        return []

    start_cols = [c for c in df.columns if _START_PATTERN.search(str(c))]
    end_cols   = [c for c in df.columns if _END_PATTERN.search(str(c))]

    if not start_cols or not end_cols:
        return []

    seen_pairs: set[tuple] = set()
    for sc in start_cols:
        for ec in end_cols:
            if sc == ec:
                continue
            pair_key = (str(sc), str(ec))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            starts = _parse_dt_silent(df[sc])
            ends   = _parse_dt_silent(df[ec])

            # Both must have at least some parseable values
            if starts.notna().sum() < 1 or ends.notna().sum() < 1:
                continue

            both_valid = starts.notna() & ends.notna()
            bad_mask   = both_valid & (starts > ends)
            count      = int(bad_mask.sum())
            if count == 0:
                continue

            # Sample: show the offending start values
            results.append(build_result(
                category      = _CAT,
                check         = "start_after_end",
                column        = str(sc),
                columns       = [str(sc), str(ec)],
                count         = count,
                total         = total,
                severity      = Severity.BAD,
                description   = (
                    f"{count:,} row(s) have '{sc}' later than '{ec}' — "
                    "a start date must not exceed its corresponding end date."
                ),
                sample_values = _sample(df.loc[bad_mask, sc]),
            ))

    return results
