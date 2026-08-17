"""Forecast view: run the real ForecastOrchestrator and visualise the timeline."""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import contracts as ct
from gui import services as svc

HORIZON_OPTIONS = [1, 3, 5, 7]


def _bounds(key):
    return ct.KPI[key]["min"], ct.KPI[key]["max"]


def render() -> None:
    c.page_title("Forecast", eyebrow="Recursive", help_text="Recursive Forecast AI (OH + NPS)")

    from gui import model_selection as ms

    c.section("Model", "\U0001f9ec")
    option = ms.render_model_selector(feature="forecast")
    family = option.family if option is not None else None

    # Scenarios: only enabled scenarios are offered (baseline deduplicated).
    scenarios = svc.list_scenarios()
    scenario_names = [s["id"] for s in scenarios]
    default_scenario = (
        ct.BASELINE_SCENARIO_ID
        if ct.BASELINE_SCENARIO_ID in scenario_names
        else (scenario_names[0] if scenario_names else ct.BASELINE_SCENARIO_ID)
    )

    st.write("")
    c.section("Forecast setup", "\U0001f39b\ufe0f")
    with st.form("forecast_form"):
        c1, c2 = st.columns(2)
        horizon_mode = c1.radio("Horizon", options=["1", "3", "5", "7", "Custom"], index=2, horizontal=True)
        horizon = int(horizon_mode) if horizon_mode != "Custom" else c1.number_input("Custom horizon (days)", 1, 365, 10)
        scenario = c2.selectbox("Scenario", options=scenario_names,
                                index=scenario_names.index(default_scenario) if default_scenario in scenario_names else 0)

        st.markdown("**Starting state (observed)**")
        q_lo, q_hi = _bounds("quality")
        cpt_lo, cpt_hi = _bounds("competency")
        a_lo, a_hi = _bounds("attendance")
        r_lo, r_hi = _bounds("release")
        t_lo, t_hi = _bounds("transfer")
        oh_lo, oh_hi = _bounds("operations_health")
        nps_lo, nps_hi = _bounds("nps")
        q1, q2, q3 = st.columns(3)
        quality = q1.number_input("Quality %", q_lo, q_hi, ct.kpi_default("quality"))
        competency = q2.number_input("Competency %", cpt_lo, cpt_hi, ct.kpi_default("competency"))
        attendance = q3.number_input("Attendance %", a_lo, a_hi, ct.kpi_default("attendance"))
        release = q1.number_input("Release Rate %", r_lo, r_hi, ct.kpi_default("release"))
        transfer = q2.number_input("Transfer Rate %", t_lo, t_hi, ct.kpi_default("transfer"))
        ops_health = q3.number_input("Operational Health %", oh_lo, oh_hi, ct.kpi_default("operations_health"))
        nps = q1.number_input(ct.NPS_INPUT_LABEL, nps_lo, nps_hi,
                              ct.kpi_default("nps"), help=ct.NPS_INPUT_HELP)

        submitted = st.form_submit_button("\u25b6  Run Forecast", type="primary",
                                          disabled=option is None, width="stretch")

    if submitted and family:
        state = {
            "quality": float(quality),
            "competency": float(competency),
            "attendance": float(attendance),
            "release": float(release),
            "transfer": float(transfer),
            "operations_health": float(ops_health),
            "nps": float(nps),
        }
        with st.spinner(f"Running H{horizon} forecast\u2026"):
            fc = c.guarded(svc.forecast, state, horizon, scenario, family)
        if fc:
            st.session_state["forecast_result"] = fc
            st.session_state["forecast_state"] = state

    from gui import charts  # noqa: E402  (function-scoped, used across result blocks)

    # ---- Result ----
    fc = st.session_state.get("forecast_result")
    if not fc:
        st.write("")
        c.empty_state("Run a forecast to see the projected OH/NPS timeline here.", icon="\U0001f4c9")
        return

    st.divider()
    if not fc.get("success"):
        c.section("Forecast failed", "\u274c")
        c.error_alert(fc.get("errors") or [])
        c.raw_json_expander(fc)
        return

    c.section(f"Forecast \u00b7 H{fc.get('horizon')} \u00b7 scenario `{fc.get('scenario')}`", "\U0001f4c8")
    ms.render_result_model(fc.get("active_family"), option)
    st.caption("Every day beyond Day 0 is a **predicted** (recursive) day — never labelled observed.")

    timeline = fc.get("timeline") or []
    if not timeline:
        st.warning("No timeline returned.")
    else:
        fig = charts.forecast_timeline_chart(timeline, fc.get("horizon") or len(timeline))
        if fig:
            st.plotly_chart(fig, width="stretch", key="legacy_forecast_view_timeline")

        with st.expander("Day-by-day table", expanded=False):
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
            st.dataframe(rows, width="stretch", hide_index=True)

    # Summary / risk / confidence / sensitivity
    summary = fc.get("summary") or {}
    if summary:
        c.section("Summary", "\U0001f9fe")
        cols = st.columns(len(summary))
        for col, (k, v) in zip(cols, summary.items()):
            with col:
                c.kpi_tile(k.replace("_", " ").title(),
                            f"{v:.3f}" if isinstance(v, (int, float)) else str(v),
                            status="none")

    risk = fc.get("risk") or {}
    confidence = fc.get("confidence") or {}
    if risk or confidence:
        with st.expander("Risk & confidence", expanded=False):
            r1, r2 = st.columns(2)
            if risk:
                r1.markdown("**Risk**")
                r1.json(risk)
            if confidence:
                r2.markdown("**Confidence**")
                r2.json(confidence)

    sensitivity = fc.get("sensitivity") or {}
    if sensitivity:
        with st.expander("Sensitivity", expanded=False):
            fig = charts.oh_sensitivity_chart(sensitivity)
            if fig:
                st.plotly_chart(fig, width="stretch", key="legacy_forecast_view_sensitivity")
            else:
                st.json(sensitivity)

    c.raw_json_expander(fc)
    c.warning_alert(fc.get("warnings") or [])

    # ---- Analytics ----
    st.divider()
    c.section("Supporting analytics", "\U0001f4d1")
    from gui.analytics import forecast as _fa
    _fa.render_analytics(st, fc)
