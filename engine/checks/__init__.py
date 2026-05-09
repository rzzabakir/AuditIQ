"""Check registry — maps category name to ordered list of inference-based callables.

Usage in aggregator:
    from engine.checks import INFERENCE_REGISTRY
    from engine.checks.schema_checks import run_schema_checks
"""
from engine.checks import (
    completeness,
    consistency,
    cross_field,
    distribution,
    format_checks,
    range_checks,
    uniqueness,
    validity,
)

# Inference-based checks (df-only, no schema required).
# Execution order within each family: coarser / more impactful checks first.
INFERENCE_REGISTRY: dict[str, list] = {
    "completeness": [
        completeness.missing_values,
        completeness.empty_rows,
        completeness.null_concentration,
    ],
    "uniqueness": [
        uniqueness.duplicate_rows,
        uniqueness.duplicate_keys_inferred,
        uniqueness.duplicate_column_combos,
    ],
    "validity": [
        validity.mixed_types,
        validity.invalid_dates,
        validity.malformed_identifiers,
    ],
    "range": [
        range_checks.future_dates,
        range_checks.impossible_values,
    ],
    "format": [
        format_checks.format_inconsistencies,
        format_checks.casing_inconsistency,
        format_checks.whitespace_issues,
    ],
    "consistency": [
        consistency.id_to_name_conflicts,
    ],
    "distribution": [
        distribution.outliers,
        distribution.value_concentration,
    ],
    "cross_field": [
        cross_field.start_after_end,
    ],
}
