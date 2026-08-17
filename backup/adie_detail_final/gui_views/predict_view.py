"""Predict view: explicit model-family selection + direct V2 prediction."""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import services as svc

DEFAULTS = {
    "quality": 87.0,
    "competency": 93.0,
    "attendance": 90.0,
    "release": 60.0,
    "transfer": 9.0,
    "operations_health": 95.0,
    "nps": 82.0,
    "total_calls_received": 2000,
}


def render() -> None:
    c.page_title("Predict", help_text="Direct prediction using the active model family")

    models = svc.list_models()
    active = svc.STATE.get_active_family()

    if not models:
        c.empty_state("No model families available. Train one on the **Train** page first.",
                      icon="🪄")
        return

    # ---- Explicit family selection ----
    options = [m["family"] for m in models if "error" not in m]
    idx = options.index(active) if active in options else 0
    family = st.selectbox("Model family (explicit)", options=options, index=idx,
                          help="Prediction uses this exact OH+NPS pair. Never mixed.")

    if family != active:
        try:
            svc.select_model_family(family)
            st.success(f"Active model family set to **{family}**")
        except Exception as exc:
            st.error(f"Could not activate {family}: {exc}")
            return

    # ---- Input form ----
    st.divider()
    with st.form("predict_form"):
        st.markdown("#### Input State")
        col1, col2, col3 = st.columns(3)
        quality = col1.number_input("Quality %", 0.0, 100.0, DEFAULTS["quality"])
        competency = col2.number_input("Competency %", 0.0, 100.0, DEFAULTS["competency"])
        attendance = col3.number_input("Attendance %", 0.0, 100.0, DEFAULTS["attendance"])
        release = col1.number_input("Release Rate %", 0.0, 100.0, DEFAULTS["release"])
        transfer = col2.number_input("Transfer Rate %", 0.0, 100.0, DEFAULTS["transfer"])
        ops_health = col3.number_input("Operational Health %", 0.0, 120.0, DEFAULTS["operations_health"])
        nps = col1.number_input("NPS", -100.0, 100.0, DEFAULTS["nps"])
        calls = col2.number_input("Total Calls Received", 1, 100000, DEFAULTS["total_calls_received"])
        submitted = st.form_submit_button("Run Prediction", type="primary")

    if submitted:
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

    # ---- Result ----
    result = st.session_state.get("predict_result")
    if result:
        st.divider()
        st.markdown("#### Prediction Result")
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
                st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Family: {result.get('active_family')} · {result.get('_timestamp', '')[:19]}")

        errors = [str(v) for v in result.get("errors", [])] if isinstance(result.get("errors"), list) else []
        if errors:
            st.error("; ".join(errors))

        c.raw_json_expander(result)


def _fmt_range(lo, hi) -> str:
    if lo is None and hi is None:
        return "—"
    return f"[{lo:.1f}, {hi:.1f}]" if lo is not None and hi is not None else str(lo or hi or "—")
