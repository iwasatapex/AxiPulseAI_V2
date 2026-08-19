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
# Semantic color tokens are exposed as root CSS variables (--ax-*) defined in
# gui.theme. Components reference them via these constants (valid CSS var()
# expressions), so every surface themes centrally instead of scattering hex.

from .theme import DEFAULT_THEME, css_variables, get_theme

ACCENT = "var(--ax-accent)"      # primary brand / active state
ACCENT_2 = "var(--ax-accent-2)"  # secondary accent
SUCCESS = "var(--ax-success)"
WARN = "var(--ax-warning)"
DANGER = "var(--ax-error)"
NEUTRAL = "var(--ax-muted)"

# Semantic status -> (color token, icon) used by status_pill() / kpi_tile().
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


def apply_theme(theme: Optional[str] = None) -> None:
    """Apply the AxiPulseAI design system for the selected theme.

    Injects the root ``--ax-*`` CSS variables from ``gui.theme`` for one of the
    10 themes (3 bright + 7 dark), the reusable component classes, and a layout
    rule that clears the Streamlit top toolbar so the application header is
    never clipped/undercut. ``theme`` is a name in ``gui.theme.THEMES``;
    defaults to ``gui.theme.DEFAULT_THEME`` (Midnight).
    """
    theme_name = theme if theme else DEFAULT_THEME
    get_theme(theme_name)  # validate/fallback
    vars_block = css_variables(theme_name)
    st.markdown(
        f"""
        <style>
        {vars_block}

        /* ---- layout: clear the Streamlit toolbar so the header is never clipped ---- */
        .block-container {{ padding-top: 2.6rem; padding-bottom: 3.5rem; max-width: 1400px; }}
        .stApp, [data-testid="stAppViewContainer"] {{ background: var(--ax-bg); }}
        [data-testid="stAppViewContainer"] {{ color: var(--ax-text); }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* ---- native Streamlit surfaces ---- */
        [data-testid="stSidebar"] {{
            background: var(--ax-sidebar-bg);
            border-right: 1px solid var(--ax-border);
        }}
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] .stMarkdown {{ color: var(--ax-text); }}
        [data-testid="stSidebar"] small, [data-testid="stCaptionContainer"] {{ color: var(--ax-muted); }}
        [data-testid="stMetricValue"] {{ font-weight: 700; color: var(--ax-text); }}
        [data-testid="stMetric"] {{
            background: var(--ax-surface-2);
            border: 1px solid var(--ax-border);
            border-radius: var(--ax-radius-sm);
            padding: 10px 14px 6px 14px;
        }}
        [data-testid="stMetricLabel"] {{ color: var(--ax-muted); }}
        [data-testid="stMetricDelta"] {{ color: var(--ax-success); }}
        hr {{ margin: 0.6rem 0 1.1rem 0 !important; opacity: 0.35; border-color: var(--ax-border); }}

        /* ---- text + headings ---- */
        html, body {{ color: var(--ax-text); background: var(--ax-bg); }}
        .stMarkdown, [data-testid="stMarkdownContainer"] {{ color: var(--ax-text); }}
        .stMarkdown p {{ line-height: 1.55; }}
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
        [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {{
            color: var(--ax-text);
        }}

        /* ---- inputs ---- */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stTimeInput"] input,
        textarea {{
            background: var(--ax-input-bg) !important;
            color: var(--ax-text) !important;
            border: 1px solid var(--ax-border) !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            background: var(--ax-input-bg) !important;
            color: var(--ax-text) !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {{ border-color: var(--ax-accent); }}

        /* ---- buttons ---- */
        [data-testid="stButton"] button,
        [data-testid="stFormSubmitButton"] button {{
            background: var(--ax-surface-2);
            color: var(--ax-text);
            border: 1px solid var(--ax-border);
        }}
        [data-testid="stButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover {{ border-color: var(--ax-accent); }}
        [data-testid="stButton"] button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {{
            background: var(--ax-accent);
            color: var(--ax-logo-text);
            border-color: var(--ax-accent);
        }}

        /* ---- sidebar navigation selection ---- */
        [data-testid="stSidebar"] [data-baseweb="radio"] label {{ color: var(--ax-text); }}
        [data-testid="stSidebar"] [role="radiogroup"] [aria-checked="true"] {{
            background: var(--ax-nav-selected);
        }}

        /* ---- alerts / info / tables ---- */
        [data-testid="stAlert"] {{ background: var(--ax-surface); border: 1px solid var(--ax-border); }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            background: var(--ax-surface); color: var(--ax-text);
        }}

        /* ---- application header (brand + status pills) ---- */
        .ap-header {{
            display: flex; align-items: center; justify-content: space-between;
            flex-wrap: wrap; gap: 12px;
            padding: 6px 0 18px 0;
            line-height: 1.4;
            min-height: 56px;
            box-sizing: border-box;
        }}
        .ap-brand {{
            font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;
            line-height: 1.25; display: flex; align-items: center; gap: 10px;
            overflow: visible;
        }}
        .ap-brand .ap-logo {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 34px; height: 34px; line-height: 1; border-radius: 8px;
            background: linear-gradient(135deg, var(--ax-accent), var(--ax-accent-2));
            color: var(--ax-logo-text); font-size: 0.95rem; font-weight: 800;
            flex-shrink: 0;
        }}
        .ap-brand small {{
            color: var(--ax-muted); font-weight: 500; font-size: 0.8rem;
            margin-left: 4px; letter-spacing: 0;
        }}
        .ap-header-meta {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
        .stMarkdown {{ overflow: visible; }}

        /* ---- cards / sections / pills ---- */
        .ap-card {{
            border: 1px solid var(--ax-border);
            border-radius: var(--ax-radius);
            padding: 16px 18px;
            margin-bottom: 12px;
            background: var(--ax-surface);
        }}
        .ap-card-title {{
            font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
            color: var(--ax-muted); font-weight: 600; margin-bottom: 6px;
        }}

        .ap-section {{
            font-size: 1.02rem; font-weight: 700; margin: 4px 0 10px 0;
            display: flex; align-items: center; gap: 8px;
        }}
        .ap-eyebrow {{
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
            color: var(--ax-muted); font-weight: 700; margin-bottom: 2px;
        }}

        .ap-pill {{
            display: inline-flex; align-items: center; gap: 5px;
            border-radius: 999px; padding: 3px 11px; font-size: 0.74rem; font-weight: 600;
            border: 1px solid currentColor;
            background: var(--ax-surface-2);
            white-space: nowrap;
        }}
        .ap-pill .dot {{ font-size: 0.6rem; }}

        .ap-model-badge {{
            display: inline-flex; flex-direction: column; gap: 2px;
            border: 1px solid var(--ax-accent);
            background: var(--ax-surface-2);
            border-radius: var(--ax-radius-sm);
            padding: 8px 12px;
        }}
        .ap-model-badge .row {{ display: flex; justify-content: space-between; gap: 14px; font-size: 0.82rem; }}
        .ap-model-badge .row .k {{ color: var(--ax-muted); }}
        .ap-model-badge .row .v {{ font-weight: 700; }}

        .ap-kpi {{
            border: 1px solid var(--ax-border);
            border-radius: var(--ax-radius);
            padding: 14px 16px;
            background: var(--ax-surface);
            height: 100%;
        }}
        .ap-kpi .label {{
            font-size: 0.76rem; color: var(--ax-muted); font-weight: 600;
            display: flex; justify-content: space-between; align-items: center;
        }}
        .ap-kpi .value {{ font-size: 1.7rem; font-weight: 800; line-height: 1.25; margin-top: 2px; }}
        .ap-kpi .sub {{ font-size: 0.78rem; color: var(--ax-muted); margin-top: 2px; }}
        .ap-kpi .trend-up {{ color: var(--ax-success); }}
        .ap-kpi .trend-down {{ color: var(--ax-error); }}
        .ap-kpi .trend-flat {{ color: var(--ax-muted); }}
        .ap-kpi .bar-track {{
            width: 100%; height: 5px; border-radius: 999px;
            background: var(--ax-surface-2); margin-top: 9px; overflow: hidden;
        }}
        .ap-kpi .bar-fill {{ height: 100%; border-radius: 999px; }}

        .ap-empty {{
            border: 1px dashed var(--ax-border);
            border-radius: var(--ax-radius);
            padding: 22px 18px; text-align: center; color: var(--ax-muted);
        }}
        .ap-empty .icon {{ font-size: 1.6rem; margin-bottom: 6px; }}
        .ap-muted {{ color: var(--ax-muted); }}
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


def status_pill(label: str, status: str = "none", sidebar: bool = False) -> None:
    """Render one semantic status pill (ready/degraded/error/etc.).

    By default it renders in the main column (Streamlit default). Pass
    ``sidebar=True`` to render it into the sidebar instead — used by the
    app's "Active model" indicator so it does not leak into the top of the
    main column above the application header.
    """
    html = _status_pill_html(label, status)
    if sidebar:
        st.sidebar.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


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
