"""Consistency checks — ID-to-name mapping conflicts."""
from __future__ import annotations

import re

import pandas as pd

from engine.checks.base import (
    Category, Severity, CheckResult,
    build_result, _sample,
)

_CAT = Category.CONSISTENCY

_ID_PATTERN = re.compile(
    r"(^|_)(id|key|code|no|num|number)($|_)",
    re.IGNORECASE,
)
_NAME_PATTERN = re.compile(
    r"(^|_)(name|title|label)($|_)",
    re.IGNORECASE,
)


def id_to_name_conflicts(df: pd.DataFrame) -> list[CheckResult]:
    """Flag cases where the same ID value maps to more than one distinct name.

    Scans heuristically detected (id_col, name_col) pairs. For each pair,
    groups by id_col and counts how many distinct values appear in name_col.
    Reports the number of id values that have conflicting names, not the
    number of rows (to give the clearest measure of magnitude).
    """
    results: list[CheckResult] = []
    total = len(df)
    if total == 0:
        return []

    id_cols   = [c for c in df.columns if _ID_PATTERN.search(str(c))]
    name_cols = [c for c in df.columns if _NAME_PATTERN.search(str(c))]
    if not id_cols or not name_cols:
        return []

    seen_pairs: set[tuple] = set()
    for id_col in id_cols:
        for name_col in name_cols:
            pair_key = tuple(sorted([str(id_col), str(name_col)]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            subset = df[[id_col, name_col]].dropna()
            if subset.empty:
                continue

            conflict_ids = (
                subset.groupby(id_col)[name_col]
                .nunique()
                .loc[lambda s: s > 1]
            )
            n_conflicts = len(conflict_ids)
            if n_conflicts == 0:
                continue

            # Report the number of conflicting id values as count
            # (total is the number of unique IDs — a ratio that makes sense)
            total_ids = subset[id_col].nunique()
            results.append(build_result(
                category      = _CAT,
                check         = "id_to_name_conflicts",
                column        = str(id_col),
                columns       = [str(id_col), str(name_col)],
                count         = n_conflicts,
                total         = total_ids,
                severity      = Severity.WARN,
                description   = (
                    f"{n_conflicts:,} value(s) in '{id_col}' map to more "
                    f"than one distinct '{name_col}' — possible data "
                    "inconsistency or merge artefact."
                ),
                sample_values = _sample(pd.Series(conflict_ids.index.tolist())),
            ))

    return results
