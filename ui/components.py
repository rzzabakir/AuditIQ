"""UI components for AuditIQ.

Each function returns an HTML string ready for `st.markdown(..., unsafe_allow_html=True)`.
Matches the CSS classes in styles/app.css.
"""

from __future__ import annotations

import html
import re
from typing import Iterable


# ── Severity helpers ──────────────────────────────────────────

# Maps each check name to its severity level.
# New checks emit severity in the result itself; this dict is the fallback
# for any result that pre-dates the new framework.
CHECK_SEVERITY: dict[str, str] = {
    # completeness
    "missing_values":                       "warn",
    "empty_rows":                           "bad",
    "null_concentration":                   "bad",
    # uniqueness
    "duplicate_rows":                       "bad",
    "duplicate_keys_inferred":              "bad",
    "duplicate_column_combos":              "warn",
    # validity
    "mixed_types":                          "warn",
    "invalid_dates":                        "warn",
    "malformed_identifiers":                "warn",
    # range
    "future_dates":                         "warn",
    "impossible_values":                    "bad",
    # format
    "format_inconsistency_name_as_value":   "info",
    "format_inconsistency_numeric_as_text": "warn",
    "format_inconsistency_date_mixed":      "warn",
    "casing_inconsistency":                 "warn",
    "whitespace_issues":                    "info",
    # consistency
    "id_to_name_conflicts":                 "warn",
    # distribution
    "outliers":                             "warn",
    "value_concentration":                  "info",
    # cross-field
    "start_after_end":                      "bad",
    # schema
    "schema_missing_column":                "bad",
    "schema_unexpected_columns":            "info",
    "schema_type_mismatch_numeric":         "warn",
    "schema_type_mismatch_date":            "warn",
    "schema_type_mismatch_text":            "warn",
    "schema_type_mismatch_identifier":      "warn",
    "schema_below_min":                     "warn",
    "schema_above_max":                     "warn",
    "schema_format_violation":              "warn",
}

# Keyed by severity level — consistent and simple.
SEVERITY_WEIGHTS: dict[str, float] = {
    "bad":  1.0,
    "warn": 0.6,
    "info": 0.2,
}


def quality_score(check_results: list[dict], total_rows: int) -> int:
    if not check_results or total_rows == 0:
        return 100
    impact = 0.0
    for r in check_results:
        # Prefer the severity embedded in the result; fall back to the lookup dict.
        sev    = r.get("severity") or CHECK_SEVERITY.get(r.get("check", ""), "info")
        weight = SEVERITY_WEIGHTS.get(sev, 0.5)
        count  = r.get("count") or 0
        finding_total = r.get("total") or total_rows
        if finding_total <= 0:
            continue
        impact += (count / finding_total) * weight
    return max(0, min(100, round(100 - impact * 100)))


# ── Small primitives ──────────────────────────────────────────

def sev_pill(level: str, text: str) -> str:
    return (
        f'<span class="pill pill-{level}">'
        f'<span class="pill-dot"></span>{html.escape(text)}'
        f'</span>'
    )


def eyebrow(text: str) -> str:
    return f'<div class="eyebrow">{html.escape(text)}</div>'


def card(inner_html: str) -> str:
    return f'<div class="card">{inner_html}</div>'


# ── Screen 1: Upload ──────────────────────────────────────────

def render_upload_header() -> str:
    return """
    <div>
      <span class="pill pill-accent"><span class="pill-dot"></span>New audit</span>
      <h1 class="hero-headline">Let's check your data.</h1>
      <p class="hero-sub">Drop a dataset in, add a schema if you have one, and we'll
      run quality checks and write up what we found.</p>
    </div>
    """


def render_dropzone_icon() -> str:
    return '<div class="dropzone-icon">↑</div>'


def render_info_pills() -> str:
    return (
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">'
        + sev_pill("info", "18 checks")
        + sev_pill("good", "Optional Gemini")
        + sev_pill("warn", "PDF report")
        + "</div>"
    )


def render_tip(title: str, body: str) -> str:
    return (
        '<div class="tip-card">'
        f'<div class="tip-card-title">{html.escape(title)}</div>'
        f'<div class="tip-card-body">{html.escape(body)}</div>'
        "</div>"
    )


# ── Screen 2: Running ─────────────────────────────────────────

def render_progress(
    stages: list[dict],
    current_index: int,
    elapsed_seconds: int,
    est_remaining_seconds: int | None,
    live_rows_scanned: int = 0,
    live_rows_total: int = 0,
    live_counters: dict[str, tuple[int, str]] | None = None,
) -> str:
    """Render the full running-state layout.

    stages: [{ "label": str, "detail": str|None }, ...]
    current_index: index of the stage currently running. Stages before it are
        rendered as done; after it as pending.
    live_counters: optional { "missing": (48, "warn"), "dupes": (12, "bad"), ... }
    """
    total = len(stages)
    pct = (current_index / total) * 100 if total else 0

    rows_html: list[str] = []
    for i, s in enumerate(stages):
        if i < current_index:
            state = "done"; icon = '<span class="stage-icon-done">✓</span>'
        elif i == current_index:
            state = "active"; icon = '<span class="stage-icon-active"></span>'
        else:
            state = "pending"; icon = '<span class="stage-icon-pending"></span>'
        label_cls = "stage-label"
        if state == "active": label_cls += " stage-label-active"
        if state == "pending": label_cls += " stage-label-pending"
        detail_html = (
            f'<div class="stage-detail">{html.escape(s["detail"])}</div>'
            if s.get("detail") else ""
        )
        now_pill = sev_pill("info", "now") if state == "active" else ""
        rows_html.append(
            f'<div class="stage-row"><div>{icon}</div>'
            f'<div style="flex:1;"><div class="{label_cls}">{html.escape(s["label"])}</div>'
            f'{detail_html}</div>{now_pill}</div>'
        )

    est_text = f"{est_remaining_seconds}s remaining" if est_remaining_seconds else "…"
    elapsed_mm = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"

    live_count_html = ""
    if live_rows_total:
        live_count_html = (
            '<div class="live-card">'
            '<div class="live-label">Live count</div>'
            f'<div class="live-big">{live_rows_scanned:,}'
            f'<span class="live-big-unit"> / {live_rows_total:,}</span></div>'
            '<div class="live-sub">rows scanned</div></div>'
        )

    counters_html = ""
    if live_counters:
        cells = []
        for label, (n, tone) in live_counters.items():
            color = {"bad": "var(--bad)", "warn": "var(--warn)"}.get(tone, "var(--ink)")
            cells.append(
                f'<div><div class="counter-big" style="color:{color};">{n}</div>'
                f'<div class="counter-label">{html.escape(label)}</div></div>'
            )
        counters_html = (
            '<div class="card"><div class="counter-grid">'
            + "".join(cells) + "</div></div>"
        )

    didyouknow_html = (
        '<div class="didyouknow">'
        '<div class="didyouknow-label">Did you know</div>'
        '<div class="didyouknow-body">Datasets with a schema file get <b>3× more precise</b> '
        'issue reports. ☕</div></div>'
    )

    return f"""
    <div class="run-header">
      <div class="run-spinner"></div>
      <div>
        <div class="run-title">Running audit…</div>
        <div class="run-sub">elapsed {elapsed_mm} · est. {est_text}</div>
      </div>
    </div>
    <div class="progressbar"><div class="progressbar-fill" style="width:{pct:.0f}%;"></div></div>
    <div style="display:grid;grid-template-columns:1.4fr 1fr;gap:24px;">
      <div class="card">
        <div class="label-uppercase">What's happening</div>
        {"".join(rows_html)}
      </div>
      <div style="display:flex;flex-direction:column;gap:14px;">
        {live_count_html}
        {counters_html}
        {didyouknow_html}
      </div>
    </div>
    """


DEFAULT_STAGES = [
    {"label": "Sipping coffee, parsing rows…",          "detail": None},
    {"label": "Squinting at types…",                    "detail": None},
    {"label": "Counting nulls on fingers…",             "detail": None},
    {"label": "Sniffing out duplicates…",               "detail": None},
    {"label": "Asking Gemini for a nice write-up",      "detail": None},
    {"label": "Printing the report",                    "detail": None},
]


# ── Screen 3: Report ─────────────────────────────────────────

def render_report_hero(filename: str, rows: int, cols: int,
                       has_schema: bool, score: int, date_str: str) -> str:
    schema_bit = "schema attached" if has_schema else "no schema"
    return f"""
    <div class="report-hero">
      <div class="report-hero-grid">
        <div style="flex:1;">
          <div class="report-hero-eyebrow">Audit report</div>
          <div class="report-hero-title">{html.escape(filename)}</div>
          <div class="report-hero-meta">{html.escape(date_str)} · {rows:,} rows × {cols} cols · {schema_bit}</div>
        </div>
        <div>
          <div class="score-chip">{score}</div>
          <div class="score-caption">quality score</div>
        </div>
      </div>
    </div>
    """


def highlight_columns_in_prose(prose: str, columns_by_severity: dict[str, str]) -> str:
    """Wrap mentions of known column names in prose with colored mono pills."""
    escaped = html.escape(prose)
    for col in sorted(columns_by_severity.keys(), key=len, reverse=True):
        level = columns_by_severity[col]
        cls = "mono-pill-bad" if level == "bad" else "mono-pill-warn"
        pattern = re.compile(rf"\b{re.escape(col)}\b")
        escaped = pattern.sub(f'<span class="{cls}">{col}</span>', escaped)
    return escaped


def _emphasise_numerics(text: str) -> str:
    return re.sub(
        r'(\d[\d,.]*(?:\s+(?:missing values|records|instances|entries|outliers|values)|\s?%))',
        r'<b style="color:var(--ink);font-weight:600;">\1</b>',
        text,
    )


def _highlight_quoted_columns(text: str, columns_by_severity: dict[str, str]) -> str:
    def _replace(m: re.Match) -> str:
        name = m.group(1)
        level = columns_by_severity.get(name, "warn")
        cls = "mono-pill-bad" if level == "bad" else "mono-pill-warn"
        return f'<span class="{cls}">{name}</span>'
    return re.sub(r"'([^']+)'", _replace, text)


def _split_lede(text: str) -> tuple[str, str]:
    idx = text.find(". ")
    if idx == -1:
        return text, ""
    return text[: idx + 1], text[idx + 2 :].strip()


def render_narrative(
    headline_sentence: str,
    paragraphs: list[str],
    columns_by_severity: dict[str, str],
) -> str:
    lede, rest = _split_lede(headline_sentence)

    # Lede: quoted-column pills only (no numeric bold in a headline)
    lede_html = _highlight_quoted_columns(html.escape(lede), columns_by_severity)

    # Body: rest of first paragraph + all subsequent narrator paragraphs
    body_parts = ([rest] if rest else []) + paragraphs
    para_htmls = []
    for p in body_parts:
        enriched = highlight_columns_in_prose(p, columns_by_severity)
        enriched = _highlight_quoted_columns(enriched, columns_by_severity)
        enriched = _emphasise_numerics(enriched)
        para_htmls.append(f"<p>{enriched}</p>")

    body_html = (
        '<div class="narrative-body">' + "".join(para_htmls) + "</div>"
        if para_htmls else ""
    )

    return (
        eyebrow("Executive summary")
        + f'<h2 class="narrative-lede">{lede_html}</h2>'
        + body_html
    )


def render_next_steps(steps: list[dict]) -> str:
    """steps: [{ "title": str, "desc": str }, ...]"""
    cards = []
    for i, s in enumerate(steps, 1):
        cards.append(
            f'<div class="action-card">'
            f'<div class="action-num">{i}</div>'
            f'<div><div class="action-title">{html.escape(s["title"])}</div>'
            f'<div class="action-desc">{html.escape(s["desc"])}</div></div>'
            f'</div>'
        )
    return f"""
    {eyebrow("What to do next")}
    <div style="display:flex;flex-direction:column;gap:10px;margin-top:10px;">
      {"".join(cards)}
    </div>
    """


def render_top_affected(rows: list[dict]) -> str:
    """rows: [{ "column": str, "pct": float, "level": "bad"|"warn" }, ...]"""
    bars = []
    for r in rows:
        bars.append(
            '<div class="topcol-row">'
            '<div class="topcol-head">'
            f'<span class="mono">{html.escape(r["column"])}</span>'
            f'<span class="mono" style="color:var(--ink-3);">{r["pct"]:.0f}%</span>'
            '</div>'
            '<div class="topcol-bar">'
            f'<div class="topcol-fill {r["level"]}" style="width:{r["pct"]:.0f}%;"></div>'
            '</div></div>'
        )
    return (
        '<div class="stat-card">'
        '<div class="label-uppercase">Top affected columns</div>'
        + "".join(bars) + "</div>"
    )


def render_donut(segments: list[tuple[str, float]], total: int) -> str:
    """segments: [(color-css-var, pct), ...] summing to 100."""
    stops, acc = [], 0.0
    for color, pct in segments:
        stops.append(f"{color} {acc}% {acc + pct}%")
        acc += pct
    gradient = f"conic-gradient({', '.join(stops)})"

    legend_rows = []
    labels = ["Format", "Missing", "Type", "Other"]
    for (color, pct), label in zip(segments, labels):
        legend_rows.append(
            f'<div class="legend-row">'
            f'<span class="legend-dot" style="background:{color};"></span>'
            f'<span style="flex:1;">{label}</span>'
            f'<span class="mono" style="color:var(--ink-3);">{pct:.0f}%</span>'
            f'</div>'
        )
    return f"""
    <div class="stat-card">
      <div class="label-uppercase" style="margin-bottom:10px;">Issue breakdown</div>
      <div style="display:flex;align-items:center;gap:14px;">
        <div class="donut-outer" style="background:{gradient};">
          <div class="donut-inner">{total}</div>
        </div>
        <div style="flex:1;font-size:12px;">{"".join(legend_rows)}</div>
      </div>
    </div>
    """


def render_compare_last(now_score: int, prev_score: int,
                        now_issues: int, prev_issues: int) -> str:
    score_up = now_score >= prev_score
    issues_down = now_issues <= prev_issues
    return f"""
    <div class="stat-card" style="background:var(--paper);">
      <div class="label-uppercase">Compare to last run</div>
      <div style="display:flex;gap:14px;margin-top:10px;">
        <div class="delta-tile">
          <div class="delta-label">Quality</div>
          <div class="delta-row">
            <span class="delta-now">{now_score}</span>
            <span class="delta-prev">{prev_score}</span>
            <span class="delta-change {'delta-up' if score_up else 'delta-down'}">
              {'↑' if score_up else '↓'} {abs(now_score - prev_score)}
            </span>
          </div>
        </div>
        <div class="delta-tile">
          <div class="delta-label">Issues</div>
          <div class="delta-row">
            <span class="delta-now">{now_issues}</span>
            <span class="delta-prev">{prev_issues}</span>
            <span class="delta-change {'delta-up' if issues_down else 'delta-down'}">
              {'↓' if issues_down else '↑'} {abs(now_issues - prev_issues)}
            </span>
          </div>
        </div>
      </div>
    </div>
    """
