"""Predict view: explicit model-family selection + direct V2 prediction."""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import contracts as ct
from gui import services as svc


def _bounds(key):
    return ct.KPI[key]["min"], ct.KPI[key]["max"]


def render() -> None:
    c.page_title("Predict", help_text="Direct prediction using an explicitly selected model")

    from gui import model_selection as ms

    option = ms.render_model_selector(feature="predict")
    family = option.family if option is not None else None

    # ---- Input form ----
    st.divider()
    with st.form("predict_form"):
        st.markdown("#### Input State")
        col1, col2, col3 = st.columns(3)
        q_lo, q_hi = _bounds("quality")
        cpt_lo, cpt_hi = _bounds("competency")
        a_lo, a_hi = _bounds("attendance")
        r_lo, r_hi = _bounds("release")
        t_lo, t_hi = _bounds("transfer")
        oh_lo, oh_hi = _bounds("operations_health")
        nps_lo, nps_hi = _bounds("nps")
        quality = col1.number_input("Quality %", q_lo, q_hi, ct.kpi_default("quality"))
        competency = col2.number_input("Competency %", cpt_lo, cpt_hi, ct.kpi_default("competency"))
        attendance = col3.number_input("Attendance %", a_lo, a_hi, ct.kpi_default("attendance"))
        release = col1.number_input("Release Rate %", r_lo, r_hi, ct.kpi_default("release"))
        transfer = col2.number_input("Transfer Rate %", t_lo, t_hi, ct.kpi_default("transfer"))
        ops_health = col3.number_input("Operational Health %", oh_lo, oh_hi, ct.kpi_default("operations_health"))
        nps = col1.number_input(ct.NPS_INPUT_LABEL, nps_lo, nps_hi,
                                ct.kpi_default("nps"), help=ct.NPS_INPUT_HELP)
        calls = col2.number_input("Total Calls Received", 1, 100000, int(ct.kpi_default("total_calls_received")))
        submitted = st.form_submit_button("Run Prediction", type="primary",
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
            "total_calls_received": float(calls),
        }
        with st.spinner("Running prediction…"):
            result = c.guarded(svc.predict, state, family)
        if result:
            st.session_state["predict_result"] = result
            st.session_state["predict_state"] = state

    # ---- Result ----
    result = st.session_state.get("predict_result")
    if result:
        st.divider()
        st.markdown("#### Prediction Result")
        ms.render_result_model(result.get("active_family"), option)
        p1, p2 = st.columns(2)
        oh = result.get("operational_health")
        nps_val = result.get("nps")
        p1.metric("Operational Health",
                  f"{oh:.1f}%" if oh is not None else "—",
                  help="Predicted operational health index")
        p2.metric("NPS",
                  f"{nps_val:.1f}" if nps_val is not None else "—",
                  help="Predicted Net Promoter Score")

        # Confidence/risk when available
        oh_conf = result.get("oh_confidence")
        nps_conf = result.get("nps_confidence")
        if oh_conf is not None or nps_conf is not None:
            st.markdown("##### Confidence")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("OH confidence", f"{oh_conf:.2f}" if oh_conf is not None else "—")
            c2.metric("OH range", _fmt_range(result.get("oh_lower"), result.get("oh_upper")))
            c3.metric("NPS confidence", f"{nps_conf:.2f}" if nps_conf is not None else "—")
            c4.metric("NPS range", _fmt_range(result.get("nps_lower"), result.get("nps_upper")))

        # NPS distribution
        dist = result.get("bayesian_score_distribution") or {}
        if dist:
            from gui import charts
            fig = charts.nps_distribution_chart(dist)
            if fig:
                st.plotly_chart(fig, width="stretch", key="predict_view_nps_distribution")

        st.caption(f"Family: {result.get('active_family')} · {result.get('_timestamp', '')[:19]}")

        errors = [str(v) for v in result.get("errors", [])] if isinstance(result.get("errors"), list) else []
        if errors:
            st.error("; ".join(errors))

        c.raw_json_expander(result)

        # ---- Analytics ----
        st.divider()
        from gui.analytics import prediction as _pa
        _pa.render_analytics(st, result, st.session_state.get("predict_state") or {})


def _fmt_range(lo, hi) -> str:
    if lo is None and hi is None:
        return "—"
    return f"[{lo:.1f}, {hi:.1f}]" if lo is not None and hi is not None else str(lo or hi or "—")
