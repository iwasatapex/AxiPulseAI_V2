"""Models view: list complete model families and select the active family."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from gui import components as c
from gui import services as svc


def render() -> None:
    c.page_title("Models", help_text="Complete OH+NPS model families")

    models = svc.list_models()
    active = svc.STATE.get_active_family()

    if not models:
        c.empty_state(
            "No complete model pairs found. "
            "A family needs both `{name}_OH.pkl` and `{name}_NPS.pkl`.",
            icon="🗂️",
        )
        return

    # ---- Active selection ----
    st.markdown("#### Select Active Model Family")
    st.caption("Only families with **both** OH and NPS files are shown. "
               "Selection is explicit — never silent.")
    options = [m["family"] for m in models if "error" not in m]
    if not options:
        c.empty_state("Model families exist but could not be inspected.", icon="⚠️")
        return
    idx = options.index(active) if active in options else 0
    chosen = st.selectbox("Model family", options=options, index=idx)
    if chosen != active:
        try:
            svc.select_model_family(chosen)
            st.success(f"Active model family set to **{chosen}**")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not activate {chosen}: {exc}")

    # ---- Table ----
    rows = []
    for m in models:
        if "error" in m:
            rows.append({"Family": m["family"], "Status": f"Error: {m['error']}"})
            continue
        oh = m.get("oh") or {}
        nps = m.get("nps") or {}
        rows.append({
            "Family": m["family"],
            "Active": "★" if m.get("active") else "",
            "OH model": oh.get("model_name"),
            "NPS model": nps.get("model_name"),
            "OH feats": oh.get("feature_count"),
            "NPS feats": nps.get("feature_count"),
            "OH hist": oh.get("history_days") if "history_days" in oh else (m.get("oh_trained_days")),
            "Saved": (m.get("saved_at") or "")[:19],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ---- Detail cards ----
    st.markdown("#### Family Details")
    for m in models:
        if "error" in m:
            continue
        with st.expander(f"{m['family']}  {'(active)' if m.get('active') else ''}", expanded=m.get("active")):
            _render_family(m)


def _render_family(m: dict) -> None:
    oh = m.get("oh") or {}
    nps = m.get("nps") or {}
    col1, col2 = st.columns(2)

    col1.markdown("**Operational Health**")
    col1.write(f"Model: {oh.get('model_name')}")
    col1.write(f"Features: {oh.get('feature_count')}")
    col1.write(f"Engine: {oh.get('engine_version')}")
    col1.write(f"Trained: {oh.get('trained')}")
    perf = oh.get("algorithm_performance")
    if perf:
        col1.caption("Algorithm performance:")
        col1.json(perf)

    col2.markdown("**NPS**")
    col2.write(f"Model: {nps.get('model_name')}")
    col2.write(f"Features: {nps.get('feature_count')}")
    col2.write(f"Engine: {nps.get('engine_version')}")
    col2.write(f"Trained: {nps.get('trained')}")
    perf = nps.get("algorithm_performance")
    if perf:
        col2.caption("Algorithm performance:")
        col2.json(perf)

    st.caption(f"Saved: {m.get('saved_at', '')[:19]}")
    c.raw_json_expander({"family": m["family"], "oh": oh, "nps": nps})
