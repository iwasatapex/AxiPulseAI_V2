"""Forecast view: run the real ForecastOrchestrator and visualise the timeline."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from gui import components as c
from gui import services as svc

HORIZON_OPTIONS = [1, 3, 5, 7]
DEFAULTS = {
    "quality": 87.0,
    "competency": 93.0,
    "attendance": 90.0,
    "release": 60.0,
    "transfer": 9.0,
    "operations_health": 95.0,
    "nps": 82.0,
}


def render() -> None:
    c.page_title("Forecast", help_text="Recursive Forecast AI (OH + NPS)")

    models = svc.list_models()
    active = svc.STATE.get_active_family()
    if not models:
        c.empty_state("No model families available. Train one first.", icon="🪄")
        return

    options = [m["family"] for m in models if "error" not in m]
    idx = options.index(active) if active in options else 0
    family = st.selectbox("Model family (explicit)", options=options, index=idx)
    if family != active:
        try:
            svc.select_model_family(family)
        except Exception as exc:
            st.error(f"Could not activate {family}: {exc}")
            return

    # Scenarios
    scenarios = svc.list_scenarios()
    scenario_names = [s["id"] for s in scenarios]
    default_scenario = "baseline" if "baseline" in scenario_names else (scenario_names[0] if scenario_names else "baseline")

    st.divider()
    with st.form("forecast_form"):
        st.markdown("#### Forecast Setup")
        c1, c2 = st.columns(2)
        horizon_mode = c1.radio("Horizon", options=["1", "3", "5", "7", "Custom"], index=2, horizontal=True)
        horizon = int(horizon_mode) if horizon_mode != "Custom" else c1.number_input("Custom horizon (days)", 1, 365, 10)
        scenario = c2.selectbox("Scenario", options=scenario_names, index=scenario_names.index(default_scenario) if default_scenario in scenario_names else 0)

        st.markdown("##### Starting State (observed)")
        q1, q2, q3 = st.columns(3)
        quality = q1.number_input("Quality %", 0.0, 100.0, DEFAULTS["quality"])
        competency = q2.number_input("Competency %", 0.0, 100.0, DEFAULTS["competency"])
        attendance = q3.number_input("Attendance %", 0.0, 100.0, DEFAULTS["attendance"])
        release = q1.number_input("Release Rate %", 0.0, 100.0, DEFAULTS["release"])
        transfer = q2.number_input("Transfer Rate %", 0.0, 100.0, DEFAULTS["transfer"])
        ops_health = q3.number_input("Operational Health %", 0.0, 120.0, DEFAULTS["operations_health"])
        nps = q1.number_input("NPS", -100.0, 100.0, DEFAULTS["nps"])

        submitted = st.form_submit_button("Run Forecast", type="primary")

    if submitted:
        state = {
            "quality": float(quality),
            "competency": float(competency),
            "attendance": float(attendance),
            "release": float(release),
            "transfer": float(transfer),
            "operations_health": float(ops_health),
            "nps": float(nps),
        }
        with st.spinner(f"Running H{horizon} forecast…"):
            fc = c.guarded(svc.forecast, state, horizon, scenario, family)
        if fc:
            st.session_state["forecast_result"] = fc

    from gui import charts  # noqa: E402  (function-scoped, used across result blocks)

    # ---- Result ----
    fc = st.session_state.get("forecast_result")
    if fc:
        st.divider()
        if not fc.get("success"):
            st.error("Forecast failed")
            c.error_alert(fc.get("errors") or [])
            c.raw_json_expander(fc)
            return

        st.markdown(f"#### Forecast · H{fc.get('horizon')} · scenario `{fc.get('scenario')}`")
        st.caption("Every day beyond Day 0 is a **predicted** (recursive) day — never labelled observed.")

        timeline = fc.get("timeline") or []
        if not timeline:
            st.warning("No timeline returned.")
        else:
            fig = charts.forecast_timeline_chart(timeline, fc.get("horizon") or len(timeline))
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            rows = []
            for i, d in enumerate(timeline):
                tag = "Day 0 (observed)" if i == 0 else f"Day {i} (predicted)"
                rows.append({
                    "Day": tag,
                    "OH": d.get("operations_health"),
                    "NPS": d.get("nps"),
                    "Quality": d.get("quality"),
                    "Comp": d.get("competency"),
                    "Attend": d.get("attendance"),
                    "Release": d.get("release"),
                    "Transfer": d.get("transfer"),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        # Summary / risk / confidence / sensitivity
        summary = fc.get("summary") or {}
        if summary:
            st.markdown("##### Summary")
            cols = st.columns(len(summary))
            for col, (k, v) in zip(cols, summary.items()):
                if isinstance(v, (int, float)):
                    col.metric(k.replace("_", " ").title(), f"{v:.3f}")
                else:
                    col.metric(k.replace("_", " ").title(), str(v))

        risk = fc.get("risk") or {}
        confidence = fc.get("confidence") or {}
        if risk or confidence:
            st.markdown("##### Risk & Confidence")
            r1, r2 = st.columns(2)
            if risk:
                r1.markdown("**Risk**")
                r1.json(risk)
            if confidence:
                r2.markdown("**Confidence**")
                r2.json(confidence)

        sensitivity = fc.get("sensitivity") or {}
        if sensitivity:
            st.markdown("##### Sensitivity")
            fig = charts.oh_sensitivity_chart(sensitivity)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.json(sensitivity)

        c.raw_json_expander(fc)

        c.warning_alert(fc.get("warnings") or [])
