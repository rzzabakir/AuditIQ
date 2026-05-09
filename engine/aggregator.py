"""Orchestrate check registry; return a flat result list."""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from engine.checks import INFERENCE_REGISTRY
from engine.checks.schema_checks import run_schema_checks

logger = logging.getLogger(__name__)


def run_all_checks(
    df: pd.DataFrame,
    schema: Optional[dict[str, Any]] = None,
    enabled_families: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Run every registered check against `df` and return a flat result list.

    Parameters
    ----------
    df:
        The dataset to audit.
    schema:
        Optional dict returned by engine.schema.load_schema().  When supplied,
        schema-driven checks are appended after inference checks.
    enabled_families:
        If given, only families in this list are run.  Defaults to all families
        in INFERENCE_REGISTRY.
    """
    if df is None:
        raise ValueError("run_all_checks received a None DataFrame")

    families = enabled_families if enabled_families is not None else list(INFERENCE_REGISTRY)
    results: list[dict[str, Any]] = []

    logger.info(
        "Running checks on %d rows × %d columns  (families: %s)",
        df.shape[0], df.shape[1], ", ".join(families),
    )

    for family in families:
        for fn in INFERENCE_REGISTRY.get(family, []):
            try:
                results.extend(fn(df))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Check '%s' failed and was skipped: %s", fn.__name__, exc
                )

    if schema:
        logger.info("Running schema-based checks")
        try:
            results.extend(run_schema_checks(schema, df))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Schema checks failed and were skipped: %s", exc)

    logger.info("Checks complete: %d findings", len(results))
    return results
