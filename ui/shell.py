"""Persistent sidebar / shell chrome for AuditIQ."""

from __future__ import annotations
import html as _html


def render_sidebar(stage: str) -> str:
    """Return HTML for the full sidebar: brand + nav + pinned footer.

    stage: 'upload' | 'running' | 'report'
    """
    upload_active = stage in ("upload", "running")
    report_active = stage == "report"

    def _nav_item(icon: str, label: str, active: bool) -> str:
        cls = "sidebar-nav-item active" if active else "sidebar-nav-item"
        return (
            f'<div class="{cls}">'
            f'<span class="sidebar-nav-icon">{icon}</span>'
            f'{_html.escape(label)}'
            f'</div>'
        )

    brand = """
    <div class="sidebar-brand">
      <div class="sidebar-brandmark">DA</div>
      <div class="sidebar-brand-text">
        <div class="sidebar-wordmark">Data Audit</div>
        <div class="sidebar-tagline">Quality checks, made readable.</div>
      </div>
    </div>
    <div class="sidebar-divider"></div>
    """

    nav = (
        '<div class="sidebar-nav">'
        + _nav_item("↑", "New audit", upload_active)
        + _nav_item("◷", "Report", report_active)
        + '</div>'
    )

    footer = """
    <div class="sidebar-footer">
      <div class="sidebar-footer-card">
        <div class="sidebar-footer-desc">AuditIQ</div>
      </div>
    </div>
    """

    return (
        '<div class="sidebar-shell">'
        + brand
        + nav
        + footer
        + '</div>'
    )
