"""Forecast view: run the real ForecastOrchestrator and visualise the timeline."""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import contracts as ct
from gui import services as svc

HORIZON_OPTIONS = [1, 3, 5, 7]


def _render_risk_human(risk) -> None:
    """Render a risk payload as human-readable text (no raw JSON).

    Accepts either a dict (with source/severity) or a scalar risk level string
    (e.g. "LOW", "ABSTAIN").  A raw dict is never shown in the primary UI — it
    stays in the technical Raw JSON expander.
    """
    if isinstance(risk, dict):
        level = risk.get("level") or risk.get("overall_risk") or risk.get("risk_level")
        source = risk.get("source")
        severity = risk.get("severity")
        if level is not None:
            st.write(f"Level: **{level}**")
        if severity is not None:
            st.write(f"Severity: **{severity}**")
        if source is not None:
            st.caption(f"Source: {source}")
        rationale = risk.get("rationale") or risk.get("reason") or risk.get("message")
        if rationale:
            st.write(str(rationale))
    else:
        # Scalar form: the canonical decision risk level (LOW/MEDIUM/HIGH/
        # ABSTAIN).  No source detail is fabricated here; a note clarifies the
        # level is a decision-layer output, not an invented severity.
        st.write(f"Risk level: **{risk}**" if risk else "Risk: —")


def _render_confidence_human(confidence) -> None:
    """Render a confidence payload as human-readable text.

    The confidence contract stamps whether the value is a business heuristic
    or calibrated/model confidence.  The wording must never claim statistical
    calibration when the value is heuristic.  All forecast confidence in this
    system is heuristic, so the scalar form is always labelled as such.
    """
    contract = {}
    score = None
    if isinstance(confidence, dict):
        contract = confidence.get("confidence_contract") or {}
        score = confidence.get("overall_confidence")
        if score is None:
            score = confidence.get("score")
    elif isinstance(confidence, (int, float)):
        # Scalar form: the ADIE package confidence value.  It is heuristic.
        score = confidence
        contract = {"kind": "heuristic", "calibrated": False, "statistical": False}

    if isinstance(score, (int, float)):
        st.write(f"Confidence: **{float(score):.2f}**")

    kind = contract.get("kind", "heuristic")
    calibrated = bool(contract.get("calibrated", False))
    if kind == "heuristic" or not calibrated:
        st.caption(
            "Heuristic confidence: deterministic weighted metric estimate, "
            "**not** calibrated statistical probability."
        )
    else:
        st.caption("Model/calibrated confidence.")

    horizon_factor = confidence.get("forecast_horizon_factor")
    if isinstance(horizon_factor, (int, float)):
        st.caption(f"Horizon decay factor: {float(horizon_factor):.2f}")


def _bounds(key):
    return ct.KPI[key]["min"], ct.KPI[key]["max"]


def render() -> None:
    c.page_title("Forecast", help_text="Recursive Forecast AI (OH + NPS)")

    from gui import model_selection as ms

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

    st.divider()
    with st.form("forecast_form"):
        st.markdown("#### Forecast Setup")
        c1, c2 = st.columns(2)
        horizon_mode = c1.radio("Horizon", options=["1", "3", "5", "7", "Custom"], index=2, horizontal=True)
        horizon = int(horizon_mode) if horizon_mode != "Custom" else c1.number_input("Custom horizon (days)", 1, 365, 10)
        scenario = c2.selectbox("Scenario", options=scenario_names, index=scenario_names.index(default_scenario) if default_scenario in scenario_names else 0)

        st.markdown("##### Starting State (observed)")
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

        submitted = st.form_submit_button("Run Forecast", type="primary",
                                          disabled=option is None)

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
        with st.spinner(f"Running H{horizon} forecast…"):
            fc = c.guarded(svc.forecast, state, horizon, scenario, family)
        if fc:
            st.session_state["forecast_result"] = fc
            st.session_state["forecast_state"] = state

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
        ms.render_result_model(fc.get("active_family"), option)
        st.caption("Every day beyond Day 0 is a **predicted** (recursive) day — never labelled observed.")

        timeline = fc.get("timeline") or []
        if not timeline:
            st.warning("No timeline returned.")
        else:
            fig = charts.forecast_timeline_chart(timeline, fc.get("horizon") or len(timeline))
            if fig:
                st.plotly_chart(fig, width="stretch", key="forecast_view_timeline")

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
            with r1:
                st.markdown("**Risk**")
                _render_risk_human(risk)
            with r2:
                st.markdown("**Confidence**")
                _render_confidence_human(confidence)

        sensitivity = fc.get("sensitivity") or {}
        if sensitivity:
            st.markdown("##### Sensitivity")
            fig = charts.oh_sensitivity_chart(sensitivity)
            if fig:
                st.plotly_chart(fig, width="stretch", key="forecast_view_sensitivity")
            else:
                st.json(sensitivity)

        c.raw_json_expander(fc)

        c.warning_alert(fc.get("warnings") or [])

        # ---- Analytics ----
        st.divider()
        from gui.analytics import forecast as _fa
        _fa.render_analytics(st, fc)
