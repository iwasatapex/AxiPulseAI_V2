"""Predict view: explicit model-family selection + direct V2 prediction."""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import contracts as ct
from gui import services as svc


def _bounds(key):
    return ct.KPI[key]["min"], ct.KPI[key]["max"]


def render() -> None:
    c.page_title("Predict", eyebrow="Interactive",
                  help_text="Direct prediction using an explicitly selected model")

    from gui import model_selection as ms

    c.section("Model", "\U0001f9ec")
    option = ms.render_model_selector(feature="predict")
    family = option.family if option is not None else None

    # ---- Input form ---------------------------------------------------
    st.write("")
    c.section("Input state", "\U0001f39b\ufe0f")
    with st.form("predict_form"):
        q_lo, q_hi = _bounds("quality")
        cpt_lo, cpt_hi = _bounds("competency")
        a_lo, a_hi = _bounds("attendance")
        r_lo, r_hi = _bounds("release")
        t_lo, t_hi = _bounds("transfer")
        oh_lo, oh_hi = _bounds("operations_health")
        nps_lo, nps_hi = _bounds("nps")

        st.markdown("**Performance drivers**")
        col1, col2, col3 = st.columns(3)
        quality = col1.number_input("Quality %", q_lo, q_hi, ct.kpi_default("quality"))
        competency = col2.number_input("Competency %", cpt_lo, cpt_hi, ct.kpi_default("competency"))
        attendance = col3.number_input("Attendance %", a_lo, a_hi, ct.kpi_default("attendance"))

        st.markdown("**Operational drivers**")
        col4, col5, col6 = st.columns(3)
        release = col4.number_input("Release Rate %", r_lo, r_hi, ct.kpi_default("release"))
        transfer = col5.number_input("Transfer Rate %", t_lo, t_hi, ct.kpi_default("transfer"))
        ops_health = col6.number_input("Operational Health %", oh_lo, oh_hi, ct.kpi_default("operations_health"))

        st.markdown("**Volume & current NPS**")
        col7, col8 = st.columns(2)
        nps = col7.number_input(ct.NPS_INPUT_LABEL, nps_lo, nps_hi,
                                ct.kpi_default("nps"), help=ct.NPS_INPUT_HELP)
        calls = col8.number_input("Total Calls Received", 1, 100000, int(ct.kpi_default("total_calls_received")))

        submitted = st.form_submit_button("\u25b6  Run Prediction", type="primary",
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
            "total_calls_received": float(calls),
        }
        with st.spinner("Running prediction\u2026"):
            result = c.guarded(svc.predict, state, family)
        if result:
            st.session_state["predict_result"] = result
            st.session_state["predict_state"] = state

    # ---- Result ---------------------------------------------------------
    result = st.session_state.get("predict_result")
    if result:
        st.divider()
        c.section("Result", "\u2705")
        ms.render_result_model(result.get("active_family"), option)

        oh = result.get("operational_health")
        nps_val = result.get("nps")
        p1, p2 = st.columns(2)
        with p1:
            c.kpi_tile("Operational Health",
                        f"{oh:.1f}%" if oh is not None else "\u2014",
                        status="ready" if oh is not None else "none",
                        help_text="Predicted operational health index")
        with p2:
            c.kpi_tile("NPS", f"{nps_val:.1f}" if nps_val is not None else "\u2014",
                        status="ready" if nps_val is not None else "none",
                        help_text="Predicted Net Promoter Score")

        # Confidence/risk when available — progressive disclosure.
        oh_conf = result.get("oh_confidence")
        nps_conf = result.get("nps_confidence")
        if oh_conf is not None or nps_conf is not None:
            with st.expander("Confidence & uncertainty", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    c.kpi_tile("OH confidence", f"{oh_conf:.2f}" if oh_conf is not None else "\u2014",
                                status="none")
                with c2:
                    c.kpi_tile("OH range", _fmt_range(result.get("oh_lower"), result.get("oh_upper")),
                                status="none")
                with c3:
                    c.kpi_tile("NPS confidence", f"{nps_conf:.2f}" if nps_conf is not None else "\u2014",
                                status="none")
                with c4:
                    c.kpi_tile("NPS range", _fmt_range(result.get("nps_lower"), result.get("nps_upper")),
                                status="none")

        # NPS distribution
        dist = result.get("bayesian_score_distribution") or {}
        if dist:
            with st.expander("Score distribution", expanded=False):
                from gui import charts
                fig = charts.nps_distribution_chart(dist)
                if fig:
                    st.plotly_chart(fig, width="stretch", key="legacy_predict_view_nps_distribution")

        st.caption(f"Family: {result.get('active_family')} \u00b7 {result.get('_timestamp', '')[:19]}")

        errors = [str(v) for v in result.get("errors", [])] if isinstance(result.get("errors"), list) else []
        if errors:
            c.error_alert(errors)

        c.raw_json_expander(result)

        # ---- Analytics ----
        st.divider()
        c.section("Contribution & explanation", "\U0001f50e")
        from gui.analytics import prediction as _pa
        _pa.render_analytics(st, result, st.session_state.get("predict_state") or {})
    else:
        st.write("")
        c.empty_state("Run a prediction to see the operational health and NPS result here.",
                      icon="\U0001f3af")


def _fmt_range(lo, hi) -> str:
    if lo is None and hi is None:
        return "\u2014"
    return f"[{lo:.1f}, {hi:.1f}]" if lo is not None and hi is not None else str(lo or hi or "\u2014")
