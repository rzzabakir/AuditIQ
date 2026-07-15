"""Build audit narratives from deterministic findings, with optional Gemini."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from engine.checks.base import display_column

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-2.5-flash-lite"
MAX_TOKENS = 2000
MAX_SAMPLE_VALUES = 5

SYSTEM_PROMPT = (
    "You are a senior data quality auditor writing a formal audit report. "
    "Translate the provided deterministic check statistics into a professional, "
    "business-readable narrative. Follow these rules strictly:\n\n"
    "1. Write in formal audit-report style using complete, flowing prose. "
    "Model the tone on this example:\n"
    "   \"The column 'Date of Birth' had 70,000 instances where values did not "
    "exist, indicating incomplete provision of information. Similarly, 94,600 "
    "instances existed where Date of Birth indicated periods after 2026, which "
    "is not possible given the submission date.\"\n"
    "2. Group findings by column. Produce exactly one paragraph per column that "
    "has at least one issue. Begin each paragraph with the column name so the "
    "reader can navigate by column.\n"
    "3. Skip columns with zero issues entirely. Do not mention them.\n"
    "4. Cite exact counts and include the percentage of the total dataset (round "
    "to one decimal place) so the reader understands magnitude.\n"
    "5. When schema notes are supplied, weave the rule naturally into the "
    "description (e.g. 'which violates the expected 13-digit identifier format').\n"
    "6. Put dataset-level findings (empty rows, duplicate rows) in a separate "
    "paragraph labelled 'Dataset-Level Issues'.\n"
    "7. Do not invent findings, hypothesise root causes, or recommend remediation. "
    "Describe only what the statistics show.\n"
    "8. Sample values are illustrative only; reference at most two or three in "
    "passing, never exhaustively.\n"
    "9. Do not use bullet lists, tables, markdown headings, or code blocks. "
    "Output flowing prose only, with a blank line between paragraphs."
)


def generate_narrative(
    check_results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    """Return an audit narrative.

    Gemini is used only when GEMINI_API_KEY is configured. The default path stays
    deterministic so a first-time user can complete an audit without credentials.
    """
    if not check_results:
        return _clean_narrative()

    payload = _build_payload(check_results, metadata)
    if not payload["columns_with_issues"] and not payload["dataset_level_issues"]:
        return _clean_narrative()

    api_key = _load_api_key()
    if not api_key:
        logger.info("GEMINI_API_KEY is not configured; using deterministic narrative.")
        return _deterministic_narrative(payload)

    user_message = (
        "Generate the audit narrative from the following aggregated statistics. "
        "Use only these numbers and the provided sample values; do not fabricate "
        "additional detail, root causes, or recommendations.\n\n"
        "```json\n" + json.dumps(payload, indent=2, default=str) + "\n```"
    )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=MODEL_ID,
        system_instruction=SYSTEM_PROMPT,
    )
    logger.info(
        "Requesting narrative (model=%s, columns=%d, dataset_level=%d)",
        MODEL_ID,
        len(payload["columns_with_issues"]),
        len(payload["dataset_level_issues"]),
    )

    try:
        response = model.generate_content(
            user_message,
            generation_config={"max_output_tokens": MAX_TOKENS},
        )
    except Exception as exc:
        logger.warning("Gemini API call failed; using deterministic narrative: %s", exc)
        return _deterministic_narrative(payload)

    narrative = (getattr(response, "text", "") or "").strip()
    if not narrative:
        logger.warning("Gemini returned an empty narrative; using deterministic narrative.")
        return _deterministic_narrative(payload)
    return narrative


def _build_payload(
    check_results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Group findings by column; strip anything that could leak raw rows."""
    by_column: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dataset_level: list[dict[str, Any]] = []

    for r in check_results:
        total = r.get("total") or 0
        count = r.get("count") or 0
        pct = round((count / total) * 100, 2) if total else 0.0

        finding: dict[str, Any] = {
            "check": r.get("check"),
            "count": count,
            "total": total,
            "percentage": pct,
            "sample_values": [
                str(v) for v in (r.get("sample_values") or [])[:MAX_SAMPLE_VALUES]
            ],
        }
        for optional_field in ("notes", "threshold", "pattern", "type_counts", "majority_type"):
            if r.get(optional_field) is not None:
                finding[optional_field] = r[optional_field]

        col = r.get("column", "")
        if col == "__row__":
            dataset_level.append(finding)
        else:
            by_column[display_column(r)].append(finding)

    return {
        "dataset": {
            "filename": metadata.get("filename"),
            "rows": metadata.get("rows"),
            "columns": metadata.get("columns"),
        },
        "columns_with_issues": [
            {"column": col, "findings": items} for col, items in by_column.items()
        ],
        "dataset_level_issues": dataset_level,
    }


def _load_api_key() -> str | None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    key = os.getenv("GEMINI_API_KEY")
    if not key or key.strip() in {"", "your_key_here"}:
        return None
    return key.strip()


def _deterministic_narrative(payload: dict[str, Any]) -> str:
    """Build a stable narrative without calling an external model."""
    paragraphs: list[str] = []
    rows = payload.get("dataset", {}).get("rows") or 0

    dataset_level = payload.get("dataset_level_issues") or []
    if dataset_level:
        issue_text = "; ".join(_finding_sentence(item, rows) for item in dataset_level)
        paragraphs.append(f"Dataset-Level Issues: {issue_text}.")

    for column_group in payload.get("columns_with_issues") or []:
        column = column_group.get("column") or "Unknown column"
        findings = column_group.get("findings") or []
        if not findings:
            continue
        issue_text = "; ".join(_finding_sentence(item, rows) for item in findings)
        paragraphs.append(f"{column}: {issue_text}.")

    if not paragraphs:
        return _clean_narrative()
    return "\n\n".join(paragraphs)


def _finding_sentence(finding: dict[str, Any], dataset_rows: int) -> str:
    check = str(finding.get("check") or "issue").replace("_", " ")
    count = finding.get("count") or 0
    total = finding.get("total") or dataset_rows
    pct = finding.get("percentage")
    if pct is None:
        pct = round((count / total) * 100, 1) if total else 0.0
    else:
        pct = round(float(pct), 1)

    sentence = f"{count:,} records ({pct:.1f}% of rows) showed {check}"
    notes = finding.get("notes")
    if notes:
        sentence += f" against the rule: {notes}"
    samples = finding.get("sample_values") or []
    if samples:
        preview = ", ".join(str(value) for value in samples[:3])
        sentence += f"; sample values include {preview}"
    return sentence


def _clean_narrative() -> str:
    return (
        "The dataset passed every deterministic data quality check performed by "
        "the audit tool. No column-level or dataset-level issues were detected."
    )
