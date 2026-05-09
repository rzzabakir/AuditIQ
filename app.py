"""Streamlit UI for AuditIQ — persistent shell + three-stage flow."""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from ai.narrator import generate_narrative
from engine.aggregator import run_all_checks
from engine.parser import load_file
from engine.schema import load_schema
from report.pdf_builder import CHECK_LABELS, generate_pdf

from ui.components import (
    CHECK_SEVERITY,
    DEFAULT_STAGES,
    quality_score,
    render_compare_last,
    render_donut,
    render_dropzone_icon,
    render_info_pills,
    render_narrative,
    render_next_steps,
    render_progress,
    render_report_hero,
    render_tip,
    render_top_affected,
    render_upload_header,
)
from ui.shell import render_sidebar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent / "samples"

# ── Page chrome ───────────────────────────────────────────────
st.set_page_config(
    page_title="AuditIQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open(Path(__file__).parent / "styles" / "app.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── State helpers ─────────────────────────────────────────────

def _init_state() -> None:
    if "stage" not in st.session_state:
        # Migrate sessions that used the old last_audit-only pattern.
        st.session_state["stage"] = (
            "report" if "last_audit" in st.session_state else "upload"
        )


def _reset_to_upload() -> None:
    for key in (
        "last_audit",
        "_pending_data_bytes", "_pending_data_name",
        "_pending_schema_bytes", "_pending_schema_name",
        "_use_sample",
    ):
        st.session_state.pop(key, None)
    st.session_state["stage"] = "upload"


def _write_tmp(data: bytes, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tmp = Path(tempfile.gettempdir()) / f"audit_{stamp}_{name}"
    tmp.write_bytes(data)
    return tmp


# ── Upload screen (stage = "upload") ─────────────────────────

def show_upload_screen() -> None:
    st.markdown(render_upload_header(), unsafe_allow_html=True)
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.3, 1], gap="large")

    with col_l:
        st.markdown(render_dropzone_icon(), unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:20px;font-weight:700;letter-spacing:-0.3px;"
            "margin-top:10px;'>Drop your dataset</div>"
            "<div style='font-size:13px;color:var(--ink-3);margin-top:2px;'>"
            ".xlsx or .csv · up to 200 MB</div>",
            unsafe_allow_html=True,
        )
        data_file = st.file_uploader(
            "Data file", type=["xlsx", "csv"], label_visibility="collapsed",
            key="data_file",
        )
        # "Browse files" lives inside st.file_uploader above; "Try sample data"
        # sits alongside it as a secondary ghost action.
        _, sample_col = st.columns([3, 2])
        with sample_col:
            try_sample = st.button(
                "Try sample data", key="try_sample", use_container_width=True,
            )
        st.markdown(render_info_pills(), unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='label-uppercase'>Optional</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-weight:600;font-size:14px;'>Schema file</div>"
            "<div style='font-size:12px;color:var(--ink-3);margin-top:2px;'>"
            "column_name, expected_type, min/max…</div>",
            unsafe_allow_html=True,
        )
        schema_file = st.file_uploader(
            "Schema", type=["xlsx", "csv"], label_visibility="collapsed",
            key="schema_file",
        )
        st.markdown(
            render_tip("TIP", "No schema? We'll infer types and flag anything suspicious."),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        run_clicked = st.button(
            "Run audit →", type="primary", use_container_width=True,
            disabled=data_file is None,
        )

    # Transitions — store pending data, advance stage, rerun so the sidebar
    # and container re-render cleanly in the new state.
    if try_sample:
        st.session_state.pop("_pending_data_bytes", None)
        st.session_state.pop("_pending_schema_bytes", None)
        st.session_state["_use_sample"] = True
        st.session_state["stage"] = "running"
        st.rerun()

    if run_clicked and data_file is not None:
        st.session_state["_pending_data_bytes"] = data_file.getvalue()
        st.session_state["_pending_data_name"] = data_file.name
        if schema_file:
            st.session_state["_pending_schema_bytes"] = schema_file.getvalue()
            st.session_state["_pending_schema_name"] = schema_file.name
        else:
            st.session_state.pop("_pending_schema_bytes", None)
            st.session_state.pop("_pending_schema_name", None)
        st.session_state["_use_sample"] = False
        st.session_state["stage"] = "running"
        st.rerun()


# ── Running screen (stage = "running") ───────────────────────

def run_audit_view() -> None:
    """Render the progress UI and execute the full audit pipeline."""
    placeholder = st.empty()
    t0 = time.time()

    def paint(
        stage_idx: int,
        rows_scanned: int = 0,
        rows_total: int = 0,
        counters: dict | None = None,
        detail_override: dict | None = None,
    ) -> None:
        stages = [dict(s) for s in DEFAULT_STAGES]
        if detail_override:
            for i, d in detail_override.items():
                stages[i]["detail"] = d
        placeholder.markdown(
            render_progress(
                stages=stages,
                current_index=stage_idx,
                elapsed_seconds=int(time.time() - t0),
                est_remaining_seconds=max(1, int((len(stages) - stage_idx) * 4)),
                live_rows_scanned=rows_scanned,
                live_rows_total=rows_total,
                live_counters=counters or {},
            ),
            unsafe_allow_html=True,
        )

    try:
        use_sample = st.session_state.get("_use_sample", False)

        if use_sample:
            data_path = SAMPLES_DIR / "Titanic-Dataset.csv"
            filename = "Titanic-Dataset.csv"
            schema_path = None
        else:
            data_bytes = st.session_state.get("_pending_data_bytes")
            if not data_bytes:
                # Shouldn't happen, but guard against a stale state.
                st.session_state["stage"] = "upload"
                st.rerun()
                return
            filename = st.session_state["_pending_data_name"]
            data_path = _write_tmp(data_bytes, filename)
            schema_bytes = st.session_state.get("_pending_schema_bytes")
            schema_path = (
                _write_tmp(schema_bytes, st.session_state["_pending_schema_name"])
                if schema_bytes else None
            )

        # Stage 0 — parse
        paint(0)
        df, metadata = load_file(data_path)
        rows_total = metadata["rows"]
        details: dict[int, str] = {0: f"{rows_total:,} rows × {metadata['columns']} cols"}

        # Stage 1 — schema
        paint(1, rows_scanned=rows_total, rows_total=rows_total, detail_override=details)
        schema = load_schema(schema_path) if schema_path is not None else None
        if schema:
            details[1] = f"schema matched {len(schema)} spec(s)"

        # Stage 2 — checks
        paint(2, rows_scanned=rows_total, rows_total=rows_total, detail_override=details)
        check_results = run_all_checks(df, schema=schema)

        counters = _counters_from_checks(check_results)
        details[2] = f"{counters.get('missing', (0, ''))[0]} missing so far"

        # Stage 3 — duplicates (already run above; just advance the visual)
        paint(3, rows_scanned=rows_total, rows_total=rows_total,
              counters=counters, detail_override=details)

        # Stage 4 — narrative
        paint(4, rows_scanned=rows_total, rows_total=rows_total,
              counters=counters, detail_override=details)
        if check_results:
            try:
                narrative = generate_narrative(check_results, metadata)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Narrative generation failed, using fallback: %s", exc)
                narrative = (
                    "Narrative generation was unavailable for this audit. "
                    "Please review the findings table above for a full breakdown of issues detected."
                )
        else:
            narrative = (
                "The dataset passed every deterministic data quality check. "
                "No issues were detected."
            )

        # Stage 5 — PDF
        paint(5, rows_scanned=rows_total, rows_total=rows_total,
              counters=counters, detail_override=details)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = Path(tempfile.gettempdir()) / f"audit_report_{stamp}.pdf"
        generate_pdf(metadata, check_results, narrative, pdf_path)
        pdf_bytes = pdf_path.read_bytes()
        pdf_path.unlink(missing_ok=True)

        st.session_state["last_audit"] = {
            "metadata": metadata,
            "check_results": check_results,
            "narrative": narrative,
            "pdf_bytes": pdf_bytes,
            "pdf_filename": f"audit_report_{stamp}.pdf",
            "has_schema": schema is not None,
            "filename": filename,
        }
        placeholder.empty()
        st.session_state["stage"] = "report"
        st.rerun()

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        placeholder.empty()
        st.session_state["stage"] = "upload"
        st.error(f"Audit failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        placeholder.empty()
        st.session_state["stage"] = "upload"
        st.error(f"Unexpected error: {exc}")
        logger.exception("Unexpected audit failure")


# ── Report screen (stage = "report") ─────────────────────────

def show_report_screen() -> None:
    audit = st.session_state["last_audit"]
    metadata = audit["metadata"]
    checks = audit["check_results"]
    score = quality_score(checks, metadata["rows"])

    st.markdown(
        render_report_hero(
            filename=audit["filename"],
            rows=metadata["rows"],
            cols=metadata["columns"],
            has_schema=audit["has_schema"],
            score=score,
            date_str=datetime.now().strftime("%b %d, %Y"),
        ),
        unsafe_allow_html=True,
    )

    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        pdf_bytes = audit.get("pdf_bytes")
        if pdf_bytes:
            st.download_button(
                "↓ Download PDF", data=pdf_bytes,
                file_name=audit.get("pdf_filename", "audit_report.pdf"),
                mime="application/pdf",
                use_container_width=True,
            )
    with c2:
        if st.button("Run new audit", use_container_width=True):
            _reset_to_upload()
            st.rerun()

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        raw = audit["narrative"].strip().split("\n\n")
        headline = raw[0] if raw else "Audit complete."
        paragraphs = raw[1:] if len(raw) > 1 else []

        columns_by_severity = _column_severity_map(checks)
        st.markdown(
            render_narrative(headline, paragraphs, columns_by_severity),
            unsafe_allow_html=True,
        )

        next_steps = _next_steps_from_checks(checks)
        if next_steps:
            st.markdown(render_next_steps(next_steps), unsafe_allow_html=True)

    with right:
        top = _top_affected(checks, metadata["rows"])
        if top:
            st.markdown(render_top_affected(top), unsafe_allow_html=True)
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        segs, total = _donut_segments(checks)
        if total:
            st.markdown(render_donut(segs, total), unsafe_allow_html=True)
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        prev = st.session_state.get("prev_audit_summary")
        if prev:
            st.markdown(
                render_compare_last(
                    now_score=score, prev_score=prev["score"],
                    now_issues=len(checks), prev_issues=prev["issues"],
                ),
                unsafe_allow_html=True,
            )
        st.session_state["prev_audit_summary"] = {"score": score, "issues": len(checks)}


# ── Shared analytics helpers ──────────────────────────────────

_SCHEMA_RANGE_CHECKS = {"schema_below_min", "schema_above_max"}


def _counters_from_checks(check_results: list[dict]) -> dict[str, tuple[int, str]]:
    buckets = {"missing": 0, "dupes": 0, "type off": 0, "out of range": 0}
    for r in check_results:
        category = r.get("category", "")
        check    = r.get("check", "")
        n        = r.get("count") or 0
        if category == "completeness":
            buckets["missing"] += n
        elif category == "uniqueness":
            buckets["dupes"] += n
        elif category == "validity":
            buckets["type off"] += n
        elif category == "range":
            buckets["out of range"] += n
        elif category == "schema" and check in _SCHEMA_RANGE_CHECKS:
            buckets["out of range"] += n
    return {
        "missing":      (buckets["missing"],      "warn"),
        "dupes":        (buckets["dupes"],         "bad"),
        "type off":     (buckets["type off"],      "warn"),
        "out of range": (buckets["out of range"],  "warn"),
    }


def _column_severity_map(checks: list[dict]) -> dict[str, str]:
    order = {"info": 0, "warn": 1, "bad": 2}
    worst: dict[str, str] = {}
    for r in checks:
        col = r.get("column", "")
        if not col or col == "__row__":
            continue
        sev = r.get("severity") or CHECK_SEVERITY.get(r.get("check", ""), "info")
        if col not in worst or order[sev] > order[worst[col]]:
            worst[col] = sev
    return worst


def _top_affected(checks: list[dict], total_rows: int, limit: int = 5) -> list[dict]:
    if not total_rows:
        return []
    by_col: dict[str, dict] = {}
    for r in checks:
        col = r.get("column", "")
        if not col or col == "__row__":
            continue
        count = r.get("count") or 0
        pct = (count / total_rows) * 100
        if col not in by_col or pct > by_col[col]["pct"]:
            sev = r.get("severity") or CHECK_SEVERITY.get(r.get("check", ""), "warn")
            by_col[col] = {"column": col, "pct": pct,
                           "level": "bad" if sev == "bad" else "warn"}
    rows = sorted(by_col.values(), key=lambda x: x["pct"], reverse=True)[:limit]
    return rows


def _donut_segments(checks: list[dict]) -> tuple[list[tuple[str, float]], int]:
    buckets = {"format": 0, "missing": 0, "type": 0, "other": 0}
    for r in checks:
        category = r.get("category", "")
        n        = r.get("count") or 0
        if category == "format":
            buckets["format"] += n
        elif category == "completeness":
            buckets["missing"] += n
        elif category == "validity":
            buckets["type"] += n
        else:
            buckets["other"] += n
    total = sum(buckets.values())
    if not total:
        return [], 0
    segs = [
        ("var(--bad)",     buckets["format"]  / total * 100),
        ("var(--accent)",  buckets["missing"] / total * 100),
        ("var(--primary)", buckets["type"]    / total * 100),
        ("var(--line)",    buckets["other"]   / total * 100),
    ]
    return segs, total


def _next_steps_from_checks(checks: list[dict]) -> list[dict]:
    worst = sorted(checks, key=lambda r: r.get("count") or 0, reverse=True)[:3]
    steps = []
    for r in worst:
        col = r.get("column", "the dataset")
        if col == "__row__":
            col = "the dataset"
        label = CHECK_LABELS.get(r.get("check", ""), r.get("check", "issue"))
        steps.append({
            "title": f"Address {label.lower()} in {col}",
            "desc": f"{r.get('count', 0):,} records affected. "
                    "Clean upstream before the next export.",
        })
    return steps


# ── Main — persistent shell + stage router ────────────────────

_init_state()
stage = st.session_state["stage"]

with st.sidebar:
    st.markdown(render_sidebar(stage), unsafe_allow_html=True)

if stage == "upload":
    show_upload_screen()
elif stage == "running":
    run_audit_view()
elif stage == "report":
    show_report_screen()
