"""Dashboard view."""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import services as svc


def render() -> None:
    c.page_title("Dashboard",
                 help_text="High-level system status for AxiPulseAI V2")

    status = svc.STATE.status()
    families = status.get("available_families") or []
    health = svc.system_health()

    # ---- System status ----
    col1, col2, col3, col4 = st.columns(4)
    active = status.get("active_family")
    col1.metric("Active Model Family", active or "—")
    col2.metric("Available Families", len(families))
    model_ok = bool(families) and (health.get("checks") or {}).get("models", {}).get("status") == "Ready"
    col3.metric("Model Status", "Ready" if model_ok else ("None" if not families else "Degraded"))
    forecast_status = health.get("status")
    col4.metric("Forecast AI", forecast_status)

    # Show a degradation detail if any dependency is not ready.
    if forecast_status != "Ready":
        details = []
        for k, chk in (health.get("checks") or {}).items():
            if chk.get("status") != "Ready":
                details.append(f"{k}: {chk.get('status')} — {chk.get('detail', chk.get('family', 'unavailable'))}")
        if details:
            st.caption("Readiness: " + "; ".join(details))

    # ---- Model availability ----
    st.markdown("#### Model Families")
    if not families:
        c.empty_state(
            "No complete model pairs (OH + NPS) found. "
            "Train one on the **Train** page first.",
            icon="🪄",
        )
    else:
        models = svc.list_models()
        rows = []
        for m in models:
            rows.append({
                "family": m["family"],
                "OH model": (m.get("oh") or {}).get("model_name"),
                "NPS model": (m.get("nps") or {}).get("model_name"),
                "OH features": (m.get("oh") or {}).get("feature_count"),
                "NPS features": (m.get("nps") or {}).get("feature_count"),
                "saved": (m.get("saved_at") or "")[:19],
                "active": "★" if m.get("active") else "",
            })
        st.dataframe(rows, width="stretch")

    # ---- Latest prediction ----
    st.markdown("#### Latest Prediction")
    last_pred = svc.STATE.get_last_prediction()
    if last_pred:
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Operational Health",
                   f"{last_pred.get('operational_health'):.1f}%" if last_pred.get("operational_health") is not None else "—")
        pc2.metric("NPS",
                   f"{last_pred.get('nps'):.1f}" if last_pred.get("nps") is not None else "—")
        pc3.metric("Family", last_pred.get("active_family") or "—")
        st.caption(f"Last prediction: {last_pred.get('_timestamp', '')[:19]}")
    else:
        c.empty_state("No prediction has been run yet. Use the **Predict** page.",
                      icon="📈")

    # ---- Latest forecast ----
    st.markdown("#### Latest Forecast")
    last_fc = svc.STATE.get_last_forecast()
    if last_fc:
        f1, f2, f3 = st.columns(3)
        f1.metric("Horizon", f"{last_fc.get('horizon')} days")
        f2.metric("Scenario", last_fc.get("scenario") or "baseline")
        f3.metric("Status", "OK" if last_fc.get("success") else "Error")
        tl = last_fc.get("timeline") or []
        if tl:
            last = tl[-1]
            st.caption(
                f"End OH {last.get('operations_health')} · End NPS {last.get('nps')} · "
                f"{last_fc.get('_timestamp', '')[:19]}"
            )
    else:
        c.empty_state("No forecast has been run yet. Use the **Forecast** page.",
                      icon="📉")

    # ---- Analytics ----
    st.divider()
    from gui.analytics import dashboard as _da
    _da.render_analytics(
        st,
        models=svc.list_models(),
        health=svc.system_health(),
        last_forecast=svc.STATE.get_last_forecast() or {},
        status=svc.STATE.status(),
    )

