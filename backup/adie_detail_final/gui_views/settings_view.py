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
    c.page_title("Settings", help_text="Project, theme, and system information")

    status = svc.STATE.status()

    st.markdown("#### Active Model Family")
    col1, col2 = st.columns(2)
    col1.metric("Active family", status.get("active_family") or "—")
    col2.metric("Provider family", status.get("provider_family") or "—")
    last_pred = status.get("last_prediction_at")
    last_fc = status.get("last_forecast_at")
    last_adie = status.get("last_adie_at")
    st.caption(
        f"Last prediction: {last_pred or '—'} · "
        f"Last forecast: {last_fc or '—'} · "
        f"Last ADIE: {last_adie or '—'}"
    )

    st.divider()
    st.markdown("#### Project Paths")
    st.write("**V2 (working):**", str(V2_ROOT))
    st.write("**V1 (frozen):**", str(V1_ROOT))
    st.write("**Training dir:**", str(V2_ROOT / "training"))
    st.write("**Models dir:**", str(V2_ROOT / "models"))

    st.divider()
    st.markdown("#### Theme")
    st.info(
        "Switch between dark/light using the **Streamlit theme selector** "
        "(menu → Settings → Theme) or edit `.streamlit/config.toml` in the "
        "V2 project root. The app styles adapt automatically via CSS variables."
    )
    st.write("**Python:**", sys.version.split()[0])
    st.write("**Platform:**", platform.platform())

    st.divider()
    st.markdown("#### About")
    st.write(
        "AxiPulseAI V2 — Healthcare CX Intelligence Engine. The GUI is a thin "
        "presentation layer over the canonical V2 services; no model, forecast, "
        "or ADIE logic is duplicated here."
    )
