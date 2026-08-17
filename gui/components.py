"""Shared Streamlit UI helpers and primitives.

Keeps presentation consistent across views and centralises the styling
used across AxiPulseAI V2 — a small, reusable design system (cards,
status pills, KPI tiles, section headers, empty/error states) so pages
compose rather than duplicate CSS/HTML.

Every function here is presentation-only. No prediction, forecast, KPI,
NPS, or OH logic lives in this module.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
# Kept as module-level constants (not just CSS vars) because several views
# pass explicit colors into components (e.g. pill(text, color=...)).

ACCENT = "#6366f1"      # indigo — primary brand / active state
ACCENT_2 = "#22d3ee"    # cyan — secondary accent, charts
SUCCESS = "#22c55e"
WARN = "#f59e0b"
DANGER = "#ef4444"
NEUTRAL = "#94a3b8"

# Semantic status -> (color, icon) used by status_pill()
_STATUS_STYLE = {
    "ready": (SUCCESS, "\u25cf"),
    "ok": (SUCCESS, "\u25cf"),
    "success": (SUCCESS, "\u25cf"),
    "active": (SUCCESS, "\u25cf"),
    "production": (SUCCESS, "\u25cf"),
    "degraded": (WARN, "\u25d0"),
    "warning": (WARN, "\u25d0"),
    "test": (WARN, "\u25d0"),
    "pending": (NEUTRAL, "\u25cb"),
    "training": (ACCENT_2, "\u25d0"),
    "none": (NEUTRAL, "\u25cb"),
    "unavailable": (DANGER, "\u25cf"),
    "error": (DANGER, "\u25cf"),
    "failed": (DANGER, "\u25cf"),
}


def apply_theme() -> None:
    """Apply the AxiPulseAI design system (CSS variables + component classes).

    Layered on top of the active Streamlit theme (dark or light) via
    Streamlit's own CSS variables, so it stays consistent across both
    without hard-coding a page background.
    """
    st.markdown(
        f"""
        <style>
        :root {{
            --ap-accent: {ACCENT};
            --ap-accent-2: {ACCENT_2};
            --ap-success: {SUCCESS};
            --ap-warn: {WARN};
            --ap-danger: {DANGER};
            --ap-neutral: {NEUTRAL};
            --ap-radius: 12px;
            --ap-radius-sm: 8px;
        }}

        .block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
        [data-testid="stSidebar"] {{ border-right: 1px solid rgba(148,163,184,0.15); }}
        [data-testid="stMetricValue"] {{ font-weight: 700; }}
        [data-testid="stMetric"] {{
            background: rgba(148,163,184,0.06);
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: var(--ap-radius-sm);
            padding: 10px 14px 6px 14px;
        }}
        hr {{ margin: 0.6rem 0 1.1rem 0 !important; opacity: 0.25; }}

        .ap-header {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 2px 0 14px 0; flex-wrap: wrap; gap: 10px;
        }}
        .ap-brand {{
            font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;
            display: flex; align-items: center; gap: 8px;
        }}
        .ap-brand .ap-logo {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 30px; height: 30px; border-radius: 8px;
            background: linear-gradient(135deg, var(--ap-accent), var(--ap-accent-2));
            color: white; font-size: 0.95rem; font-weight: 800;
        }}
        .ap-brand small {{
            color: var(--ap-neutral); font-weight: 500; font-size: 0.8rem;
            margin-left: 4px; letter-spacing: 0;
        }}
        .ap-header-meta {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}

        .ap-card {{
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: var(--ap-radius);
            padding: 16px 18px;
            margin-bottom: 12px;
            background: rgba(148,163,184,0.045);
        }}
        .ap-card-title {{
            font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
            color: var(--ap-neutral); font-weight: 600; margin-bottom: 6px;
        }}

        .ap-section {{
            font-size: 1.02rem; font-weight: 700; margin: 4px 0 10px 0;
            display: flex; align-items: center; gap: 8px;
        }}
        .ap-eyebrow {{
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
            color: var(--ap-neutral); font-weight: 700; margin-bottom: 2px;
        }}

        .ap-pill {{
            display: inline-flex; align-items: center; gap: 5px;
            border-radius: 999px; padding: 3px 11px; font-size: 0.74rem; font-weight: 600;
            border: 1px solid currentColor;
            background: rgba(148,163,184,0.1);
            white-space: nowrap;
        }}
        .ap-pill .dot {{ font-size: 0.6rem; }}

        .ap-model-badge {{
            display: inline-flex; flex-direction: column; gap: 2px;
            border: 1px solid rgba(99,102,241,0.35);
            background: rgba(99,102,241,0.08);
            border-radius: var(--ap-radius-sm);
            padding: 8px 12px;
        }}
        .ap-model-badge .row {{
            display: flex; justify-content: space-between; gap: 14px; font-size: 0.82rem;
        }}
        .ap-model-badge .row .k {{ color: var(--ap-neutral); }}
        .ap-model-badge .row .v {{ font-weight: 700; }}

        .ap-kpi {{
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: var(--ap-radius);
            padding: 14px 16px;
            background: rgba(148,163,184,0.045);
            height: 100%;
        }}
        .ap-kpi .label {{
            font-size: 0.76rem; color: var(--ap-neutral); font-weight: 600;
            display: flex; justify-content: space-between; align-items: center;
        }}
        .ap-kpi .value {{ font-size: 1.7rem; font-weight: 800; line-height: 1.25; margin-top: 2px; }}
        .ap-kpi .sub {{ font-size: 0.78rem; color: var(--ap-neutral); margin-top: 2px; }}
        .ap-kpi .trend-up {{ color: var(--ap-success); }}
        .ap-kpi .trend-down {{ color: var(--ap-danger); }}
        .ap-kpi .trend-flat {{ color: var(--ap-neutral); }}
        .ap-kpi .bar-track {{
            width: 100%; height: 5px; border-radius: 999px;
            background: rgba(148,163,184,0.18); margin-top: 9px; overflow: hidden;
        }}
        .ap-kpi .bar-fill {{ height: 100%; border-radius: 999px; }}

        .ap-empty {{
            border: 1px dashed rgba(148,163,184,0.3);
            border-radius: var(--ap-radius);
            padding: 22px 18px; text-align: center; color: var(--ap-neutral);
        }}
        .ap-empty .icon {{ font-size: 1.6rem; margin-bottom: 6px; }}
        .ap-muted {{ color: var(--ap-neutral); }}
        .ap-code {{ font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.85rem; }}

        button[data-baseweb="tab"] {{ font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_pill_html(label: str, status: str) -> str:
    color, icon = _STATUS_STYLE.get((status or "").lower(), (NEUTRAL, "\u25cb"))
    return (
        f'<span class="ap-pill" style="color:{color}">'
        f'<span class="dot">{icon}</span>{label}</span>'
    )


def brand_header(subtitle: Optional[str] = None, meta: Optional[List[Dict[str, str]]] = None) -> None:
    """Application header: logo mark, product name, subtitle, and optional
    compact status meta (e.g. active model family, system health) rendered
    as pills on the right.

    ``meta`` items: {"label": str, "status": "ready"|"warning"|"error"|"none"|...}
    """
    meta_html = ""
    if meta:
        chips = "".join(_status_pill_html(m.get("label", ""), m.get("status", "none")) for m in meta)
        meta_html = f'<div class="ap-header-meta">{chips}</div>'

    st.markdown(
        f'''
        <div class="ap-header">
            <div class="ap-brand">
                <span class="ap-logo">AI</span>
                AxiPulse<span style="color:{ACCENT}">AI</span>
                <small>V2 &middot; Command Center</small>
            </div>
            {meta_html}
        </div>
        ''',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)
    st.divider()


def status_pill(label: str, status: str = "none") -> None:
    """Render one semantic status pill (ready/degraded/error/etc.)."""
    st.markdown(_status_pill_html(label, status), unsafe_allow_html=True)


def page_title(title: str, help_text: Optional[str] = None, eyebrow: Optional[str] = None) -> None:
    if eyebrow:
        st.markdown(f'<div class="ap-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    cols = st.columns([4, 1])
    cols[0].markdown(f"### {title}")
    if help_text:
        cols[1].markdown(
            f"<div class='ap-muted' style='text-align:right;padding-top:10px'>{help_text}</div>",
            unsafe_allow_html=True,
        )


def section(title: str, icon: str = "") -> None:
    """A compact section heading used to break a page into zones
    (context → primary decision → supporting analytics → diagnostics)."""
    icon_html = f'<span>{icon}</span>' if icon else ""
    st.markdown(f'<div class="ap-section">{icon_html}{title}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: Any, help_text: Optional[str] = None,
                delta: Any = None) -> None:
    st.metric(label=label, value=value, delta=delta, help=help_text)


def kpi_tile(label: str, value: str, target: Optional[str] = None,
             gap: Optional[str] = None, trend: Optional[str] = None,
             status: str = "none", help_text: Optional[str] = None,
             progress: Optional[float] = None) -> None:
    """Rich KPI card: value + target + gap + trend + status, with an
    optional progress-to-target bar. ``progress`` is 0..1 (clamped).

    ``trend`` is a short pre-formatted string (e.g. "+2.1%"); this only
    controls color (green if it starts with '+', red if '-'), it never
    computes or reinterprets the figure.
    """
    color, icon = _STATUS_STYLE.get((status or "").lower(), (NEUTRAL, "\u25cb"))
    trend_html = ""
    if trend:
        cls = "trend-flat"
        if trend.strip().startswith("+"):
            cls = "trend-up"
        elif trend.strip().startswith("-"):
            cls = "trend-down"
        trend_html = f'<span class="{cls}">{trend}</span>'

    sub_bits = []
    if target:
        sub_bits.append(f"Target {target}")
    if gap:
        sub_bits.append(f"Gap {gap}")
    sub_html = " &middot; ".join(sub_bits)
    if trend_html:
        sub_html = f"{sub_html} &middot; {trend_html}" if sub_html else trend_html

    bar_html = ""
    if progress is not None:
        pct = max(0.0, min(1.0, float(progress))) * 100
        bar_html = (
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{pct:.0f}%;background:{color}"></div></div>'
        )

    title_attr = f' title="{help_text}"' if help_text else ""
    st.markdown(
        f'''
        <div class="ap-kpi"{title_attr}>
            <div class="label">
                <span>{label}</span>
                <span style="color:{color}">{icon}</span>
            </div>
            <div class="value">{value}</div>
            <div class="sub">{sub_html}&nbsp;</div>
            {bar_html}
        </div>
        ''',
        unsafe_allow_html=True,
    )


def model_badge(family: str, oh_algo: Optional[str] = None, nps_algo: Optional[str] = None,
                 status: str = "none") -> None:
    """The 'active model, everywhere' badge — used in header/sidebar/selector
    so the currently selected family is never ambiguous."""
    color, icon = _STATUS_STYLE.get((status or "").lower(), (NEUTRAL, "\u25cb"))
    rows = f'<div class="row"><span class="k">Family</span><span class="v">{family}</span></div>'
    if oh_algo:
        rows += f'<div class="row"><span class="k">OH</span><span class="v">{oh_algo}</span></div>'
    if nps_algo:
        rows += f'<div class="row"><span class="k">NPS</span><span class="v">{nps_algo}</span></div>'
    rows += (
        f'<div class="row"><span class="k">Status</span>'
        f'<span class="v" style="color:{color}">{icon} {status.title()}</span></div>'
    )
    st.markdown(f'<div class="ap-model-badge">{rows}</div>', unsafe_allow_html=True)


def pill(text: str, color: str = NEUTRAL) -> None:
    st.markdown(f'<span class="ap-pill" style="color:{color}">{text}</span>',
                unsafe_allow_html=True)


def raw_json_expander(data: Any, label: str = "Raw JSON") -> None:
    """Collapsible raw structured output (never fabricated)."""
    with st.expander(label, expanded=False):
        try:
            st.json(json.loads(json.dumps(data, default=str)))
        except Exception:
            st.code(str(data))


def empty_state(message: str, icon: str = "\U0001f5c2\ufe0f") -> None:
    st.markdown(
        f"<div class='ap-empty'><div class='icon'>{icon}</div>"
        f"<div>{message}</div></div>",
        unsafe_allow_html=True,
    )


def error_alert(errors: List[str]) -> None:
    for err in errors:
        st.error(str(err))


def warning_alert(warnings: List[str]) -> None:
    for w in warnings:
        st.warning(str(w))


def guarded(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a callable and surface exceptions as a Streamlit error."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - GUI boundary
        st.error(f"{type(exc).__name__}: {exc}")
        return None


def family_selector(label: str = "Model family",
                    options: Optional[List[str]] = None) -> Optional[str]:
    """Render a select box for the active model family (explicit, never silent)."""
    if options is None:
        from gui import services as svc
        options = [m["family"] for m in svc.list_models()]
    if not options:
        st.warning("No complete model families available. Train one on the **Train** page first.")
        return None
    return st.selectbox(label, options=options, index=0)
