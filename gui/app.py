"""AxiPulseAI V2 — Streamlit GUI entry point.

Run from the V2 project root:
    source .venv/bin/activate
    python -m streamlit run gui/app.py
    # or without activating the venv:
    .venv/bin/python -m streamlit run gui/app.py
    # or the convenience launcher:
    ./gui/run.sh
"""
from __future__ import annotations

import sys


def _check_runtime_deps() -> None:
    """Fail fast with an actionable message if the GUI's core deps are absent.

    The service layer transitively imports ``core.probabilistic.result`` which
    requires Pydantic 2.x (``BaseModel`` / ``field_validator`` /
    ``model_validator``). If the app is launched with the wrong interpreter
    (e.g. a system Python that happens to have ``streamlit`` but no
    ``pydantic``), the raw traceback is confusing — surface the fix instead.
    """
    problems: list[str] = []
    try:
        import pydantic
    except ImportError:
        problems.append("pydantic (2.x required)")
    else:
        if int(pydantic.VERSION.split(".")[0]) < 2:
            problems.append(f"pydantic {pydantic.VERSION} (2.x required)")
    try:
        import streamlit  # noqa: F401
    except ImportError:
        problems.append("streamlit")
    if problems:
        print(
            "AxiPulseAI V2 GUI cannot start — missing/unsupported runtime "
            f"dependencies: {', '.join(problems)}.\n"
            "Use the project virtualenv, e.g.:\n"
            "    source .venv/bin/activate\n"
            "    python -m streamlit run gui/app.py\n"
            "or install the pinned environment:\n"
            "    python -m pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(1)


_check_runtime_deps()

import streamlit as st

from gui import components as c
from gui import services as svc
from gui.views import (
    adie_view,
    dashboard_view,
    forecast_view,
    models_view,
    predict_view,
    reverse_view,
    settings_view,
    target_state_view,
    train_view,
)

NAV = {
    "Dashboard": dashboard_view,
    "Train": train_view,
    "Models": models_view,
    "Predict": predict_view,
    "Forecast": forecast_view,
    "Target State": target_state_view,
    "Reverse Optimizer": reverse_view,
    "ADIE Decision": adie_view,
    "Settings": settings_view,
}

st.set_page_config(
    page_title="AxiPulseAI V2",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


_NAV_ICONS = {
    "Dashboard": "\U0001f4ca",
    "Train": "\U0001f9e0",
    "Models": "\U0001f9ec",
    "Predict": "\U0001f3af",
    "Forecast": "\U0001f4c8",
    "Target State": "\U0001f6a9",
    "Reverse Optimizer": "\U0001f501",
    "ADIE Decision": "\u2696\ufe0f",
    "Settings": "\u2699\ufe0f",
}


def main() -> None:
    # Read the persisted theme (defaults to Midnight = the current dark look)
    # and apply the design system up-front so every surface themes immediately.
    from gui.theme import DEFAULT_THEME, theme_names

    theme = svc.STATE.get_theme() or DEFAULT_THEME
    c.apply_theme(theme)

    status = svc.STATE.status()
    active = status.get("active_family")
    health = svc.system_health()

    st.sidebar.markdown(
        '<div class="ap-brand" style="font-size:1.1rem">'
        '<span class="ap-logo">AI</span>AxiPulse'
        '<span style="color:var(--ax-accent)">AI</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Healthcare CX Intelligence")
    st.sidebar.divider()

    # Current active family shown in the sidebar for clarity — the active
    # model is never silently changed, only ever reflected here.
    if active:
        c.status_pill(f"Active: {active}", "active", sidebar=True)
    else:
        c.status_pill("No active model", "none", sidebar=True)
        st.sidebar.caption("Select one on **Models**.")

    # Persistent appearance selector (10 themes: 3 bright + 7 dark).
    _render_theme_selector(theme)

    # Honor cross-page navigation requests (e.g. a zero-model state offering
    # a "Go to Train page" action). The widget key is set before the radio is
    # instantiated so it becomes the radio's initial value on this rerun.
    target = st.session_state.pop("apgui_go", None)
    if target in NAV:
        st.session_state["apgui_page"] = target

    page = st.sidebar.radio(
        "Navigate",
        options=list(NAV.keys()),
        format_func=lambda p: f"{_NAV_ICONS.get(p, '')}  {p}",
        label_visibility="collapsed",
        key="apgui_page",
    )

    st.sidebar.divider()
    st.sidebar.caption("GUI → V2 services → Models / Forecast AI → ADIE V3")

    header_meta = [{"label": page, "status": "none"}]
    if active:
        header_meta.append({"label": f"Model: {active}", "status": "ready"})
    else:
        header_meta.append({"label": "No model", "status": "none"})
    header_meta.append({"label": f"System: {health.get('status', 'Unknown')}",
                         "status": "ready" if health.get("status") == "Ready" else "degraded"})

    c.brand_header("Operational Health & NPS Decision Intelligence", meta=header_meta)

    # Dispatch to the selected view module.
    view = NAV[page]
    view.render()


def _render_theme_selector(current: str) -> None:
    """Sidebar appearance selector; persists the chosen theme in session state.

    Changing the theme immediately re-applies ``apply_theme`` (which runs at
    the top of ``main`` on the rerun) and survives navigation between pages.
    """
    from gui.theme import DEFAULT_THEME, THEMES, theme_names

    options = theme_names()
    current = svc.STATE.get_theme() or DEFAULT_THEME
    idx = options.index(current) if current in options else options.index(DEFAULT_THEME)

    with st.sidebar.expander("Appearance", expanded=False):
        chosen = st.selectbox(
            "Theme",
            options=options,
            index=idx,
            format_func=lambda name: name,
            key="apgui_theme_picker",
            help=f"App theme ({len(THEMES)} available: 3 bright + 7 dark).",
        )
        if chosen and chosen != current:
            svc.STATE.set_theme(chosen)
            st.rerun()


if __name__ == "__main__":
    main()
