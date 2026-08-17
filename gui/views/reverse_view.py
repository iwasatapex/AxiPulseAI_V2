"""Reverse Optimizer view: find the KPIs that drive a single target metric."""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import contracts as ct
from gui import services as svc

STATE_LABELS = {
    "quality": "Quality",
    "competency": "Competency",
    "attendance": "Attendance",
    "release": "Release Rate",
    "transfer": "Transfer Rate",
    "total_calls_received": "Total Calls Received",
    "operational_health": "Operational Health",
    "nps": "NPS",
}


def render() -> None:
    c.page_title("Reverse Optimizer", help_text="Reverse-optimise KPIs for a target")

    from gui import model_selection as ms

    option = ms.render_model_selector(feature="reverse")
    family = option.family if option is not None else None

    st.markdown(
        "Pick a metric and a target value; the engine finds the operational "
        "KPIs that get you closest to it. Delegates to the canonical "
        "**TargetStateEngine** — expect a minute or two per run."
    )

    metric = st.radio("Metric to optimise", options=["OH", "NPS"], horizontal=True,
                      help="OH = Operational Health, NPS = Net Promoter Score")

    bounds = {"OH": (ct.OH_MIN, ct.OH_MAX), "NPS": (ct.NPS_MIN, ct.NPS_MAX)}
    lo, hi = bounds[metric]
    default = ct.kpi_default("operations_health") if metric == "OH" else ct.kpi_default("nps")

    with st.form("reverse_opt_form"):
        target = st.number_input(f"Target {metric}", lo, hi, default,
                                 step=1.0,
                                 help=f"Desired {'operational health' if metric == 'OH' else 'NPS'} value "
                                      f"({lo:g}–{hi:g}).")
        submitted = st.form_submit_button("Run Reverse Optimization", type="primary",
                                          disabled=option is None)

    if submitted and family:
        with st.spinner(f"Searching KPIs that achieve target {metric} = {target}…"):
            result = c.guarded(svc.reverse_optimize, metric, float(target), family=family)
        if result:
            st.session_state["reverse_result"] = result

    result = st.session_state.get("reverse_result")
    if not result:
        return

    st.divider()
    ms.render_result_model(result.get("active_family"), option)
    st.markdown(f"#### Recommended KPIs for {metric} = {result.get('target')}")

    predicted = result.get("predicted")
    distance = result.get("distance")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Predicted {metric}", f"{predicted:.2f}" if predicted is not None else "—")
    m2.metric("Distance from target",
              f"{distance:.3f}" if distance is not None else "—")
    m3.metric("Target", f"{result.get('target')}")

    rec = result.get("recommended_state") or {}
    if not result.get("found") or not rec:
        st.error("No KPI combination found for this target.")
    else:
        rows = []
        for key, label in STATE_LABELS.items():
            if key in rec:
                rows.append({"KPI": label, "Value": rec[key]})
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)

    st.caption("Consensus predictions: " + str(result.get("consensus") or {}))
    c.raw_json_expander(result)

    # ---- Analytics ----
    st.divider()
    from gui.analytics import reverse as _ra
    _ra.render_analytics(st, result)
