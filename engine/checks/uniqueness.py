"""Uniqueness checks — duplicate rows, inferred duplicate keys, combo dupes."""
from __future__ import annotations

import re

import pandas as pd

from engine.checks.base import (
    Category, Severity, CheckResult,
    build_result, _sample,
)

_CAT = Category.UNIQUENESS

# Regex to detect ID-like column names (word-boundary anchored).
_ID_PATTERN = re.compile(
    r"(^|_)(id|key|code|no|num|number)($|_)",
    re.IGNORECASE,
)
_NAME_PATTERN = re.compile(
    r"(^|_)(name|title|label|desc|description)($|_)",
    re.IGNORECASE,
)
_KEY_UNIQUENESS_MIN = 0.70   # column must be ≥70% unique to be treated as an ID key


def duplicate_rows(df: pd.DataFrame) -> list[CheckResult]:
    """Exact full-row duplicates (keeps first occurrence)."""
    total = len(df)
    if total == 0:
        return []
    mask = df.duplicated(keep="first")
    count = int(mask.sum())
    if count == 0:
        return []
    idx_samples = [f"row_index={i}" for i in mask[mask].index[:5].tolist()]
    return [build_result(
        category      = _CAT,
        check         = "duplicate_rows",
        column        = "__row__",
        count         = count,
        total         = total,
        severity      = Severity.BAD,
        description   = f"{count:,} rows are exact duplicates of an earlier row.",
        sample_values = idx_samples,
    )]


def duplicate_keys_inferred(df: pd.DataFrame) -> list[CheckResult]:
    """Detect duplicate values in columns that look like unique identifiers.

    A column qualifies as a candidate key if:
    - Its name matches the ID_PATTERN (e.g. customer_id, branch_code, account_no)
    - At least KEY_UNIQUENESS_MIN fraction of its non-null values are distinct
      (avoids flagging low-cardinality category codes)
    """
    results: list[CheckResult] = []
    total = len(df)
    if total == 0:
        return []

    for col in df.columns:
        col_str = str(col)
        if not _ID_PATTERN.search(col_str):
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        uniqueness = series.nunique() / len(series)
        if uniqueness < _KEY_UNIQUENESS_MIN:
            continue
        dup_mask = df[col].duplicated(keep="first") & df[col].notna()
        count = int(dup_mask.sum())
        if count == 0:
            continue
        results.append(build_result(
            category      = _CAT,
            check         = "duplicate_keys_inferred",
            column        = col_str,
            count         = count,
            total         = total,
            severity      = Severity.BAD,
            description   = (
                f"{count:,} rows contain a duplicate value in '{col_str}', "
                "which appears to be a unique identifier column."
            ),
            sample_values = _sample(df.loc[dup_mask, col]),
        ))
    return results


def duplicate_column_combos(df: pd.DataFrame) -> list[CheckResult]:
    """Flag rows that repeat the same (id_col, name_col) value combination.

    Searches for heuristically paired ID + name columns and reports cases
    where the same ID appears with more than one distinct name (a consistency
    signal) OR where the same id+name pair repeats verbatim (a duplication
    signal).  Only the verbatim-repetition case is reported here; the
    id→multiple-names case is covered by consistency.id_to_name_conflicts.
    """
    results: list[CheckResult] = []
    total = len(df)
    if total == 0:
        return []

    id_cols   = [c for c in df.columns if _ID_PATTERN.search(str(c))]
    name_cols = [c for c in df.columns if _NAME_PATTERN.search(str(c))]

    seen_pairs: set[tuple[str, str]] = set()
    for id_col in id_cols:
        for name_col in name_cols:
            pair = tuple(sorted([str(id_col), str(name_col)]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            combo_cols = [id_col, name_col]
            if any(c not in df.columns for c in combo_cols):
                continue
            mask = df.duplicated(subset=combo_cols, keep="first")
            count = int(mask.sum())
            if count == 0:
                continue
            col_label = f"{id_col} + {name_col}"
            results.append(build_result(
                category    = _CAT,
                check       = "duplicate_column_combos",
                column      = "__multi__",
                columns     = list(combo_cols),
                count       = count,
                total       = total,
                severity    = Severity.WARN,
                description = (
                    f"{count:,} rows repeat an identical "
                    f"({col_label}) value combination."
                ),
            ))
    return results
