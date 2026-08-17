"""Settings view: system info, theme guidance, project paths."""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import streamlit as st

from gui import components as c
from gui import services as svc

V2_ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = Path(__file__).resolve().parents[2] / "AxiPulseAI_V1"


def render() -> None:
    c.page_title("Settings", eyebrow="System", help_text="Project, theme, and system information")

    status = svc.STATE.status()

    c.section("Active model family", "\U0001f3af")
    col1, col2 = st.columns(2)
    with col1:
        c.kpi_tile("Active family", status.get("active_family") or "\u2014",
                    status="active" if status.get("active_family") else "none")
    with col2:
        c.kpi_tile("Provider family", status.get("provider_family") or "\u2014",
                    status="ready" if status.get("provider_family") else "none")
    last_pred = status.get("last_prediction_at")
    last_fc = status.get("last_forecast_at")
    last_adie = status.get("last_adie_at")
    st.caption(
        f"Last prediction: {last_pred or '\u2014'} \u00b7 "
        f"Last forecast: {last_fc or '\u2014'} \u00b7 "
        f"Last ADIE: {last_adie or '\u2014'}"
    )

    st.divider()
    c.section("Project paths", "\U0001f4c1")
    with st.expander("Paths", expanded=True):
        st.write("**V2 (working):**", str(V2_ROOT))
        st.write("**V1 (frozen):**", str(V1_ROOT))
        st.write("**Training dir:**", str(V2_ROOT / "training"))
        st.write("**Models dir:**", str(V2_ROOT / "models"))

    st.divider()
    c.section("Theme", "\U0001f3a8")
    st.info(
        "Switch between dark/light using the **Streamlit theme selector** "
        "(menu → Settings → Theme) or edit `.streamlit/config.toml` in the "
        "V2 project root. The app styles adapt automatically via CSS variables."
    )
    scol1, scol2 = st.columns(2)
    with scol1:
        c.kpi_tile("Python", sys.version.split()[0], status="none")
    with scol2:
        c.kpi_tile("Platform", platform.platform(), status="none")

    st.divider()
    c.section("About", "\u2139\ufe0f")
    st.write(
        "AxiPulseAI V2 — Healthcare CX Intelligence Engine. The GUI is a thin "
        "presentation layer over the canonical V2 services; no model, forecast, "
        "or ADIE logic is duplicated here."
    )
