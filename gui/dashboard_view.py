"""Dashboard view."""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import services as svc


def render() -> None:
    c.page_title(
        "Dashboard",
        eyebrow="Overview",
        help_text="High-level system status for AxiPulseAI V2",
    )

    status = svc.STATE.status()
    families = status.get("available_families") or []
    health = svc.system_health()
    active = status.get("active_family")

    # ---- Context row: model + data + system status -----------------------
    c.section("System status", "\U0001f9ed")
    model_ok = bool(families) and (health.get("checks") or {}).get("models", {}).get("status") == "Ready"
    model_status = "ready" if model_ok else ("none" if not families else "degraded")
    forecast_status = health.get("status")
    fc_status_key="legacy_ready" if forecast_status == "Ready" else "degraded"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        c.kpi_tile("Active Model Family", active or "\u2014", status=model_status)
    with k2:
        c.kpi_tile("Available Families", str(len(families)),
                    status="ready" if families else "none")
    with k3:
        c.kpi_tile("Model Status", "Ready" if model_ok else ("None" if not families else "Degraded"),
                    status=model_status)
    with k4:
        c.kpi_tile("Forecast AI", str(forecast_status), status=fc_status_key)

    # Show a degradation detail if any dependency is not ready.
    if forecast_status != "Ready":
        details = []
        for k, chk in (health.get("checks") or {}).items():
            if chk.get("status") != "Ready":
                details.append(f"{k}: {chk.get('status')} — {chk.get('detail', chk.get('family', 'unavailable'))}")
        if details:
            st.caption("Readiness: " + "; ".join(details))

    st.write("")

    # ---- Model availability -----------------------------------------------
    c.section("Model families", "\U0001f9ec")
    if not families:
        c.empty_state(
            "No complete model pairs (OH + NPS) found. "
            "Train one on the **Train** page first.",
            icon="\U0001fa84",
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
                "active": "\u2605" if m.get("active") else "",
            })
        st.dataframe(rows, width="stretch", hide_index=True)

    st.write("")

    # ---- Latest prediction + forecast, side by side -----------------------
    col_pred, col_fc = st.columns(2)

    with col_pred:
        c.section("Latest prediction", "\U0001f3af")
        last_pred = svc.STATE.get_last_prediction()
        if last_pred:
            oh_val = last_pred.get("operational_health")
            nps_val = last_pred.get("nps")
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                c.kpi_tile("Operational Health",
                            f"{oh_val:.1f}%" if oh_val is not None else "\u2014",
                            status="ready" if oh_val is not None else "none")
            with pc2:
                c.kpi_tile("NPS", f"{nps_val:.1f}" if nps_val is not None else "\u2014",
                            status="ready" if nps_val is not None else "none")
            with pc3:
                c.kpi_tile("Family", last_pred.get("active_family") or "\u2014", status="none")
            st.caption(f"Last prediction: {last_pred.get('_timestamp', '')[:19]}")
        else:
            c.empty_state("No prediction has been run yet. Use the **Predict** page.",
                          icon="\U0001f4c8")

    with col_fc:
        c.section("Latest forecast", "\U0001f4c9")
        last_fc = svc.STATE.get_last_forecast()
        if last_fc:
            f1, f2, f3 = st.columns(3)
            with f1:
                c.kpi_tile("Horizon", f"{last_fc.get('horizon')} days", status="none")
            with f2:
                c.kpi_tile("Scenario", str(last_fc.get("scenario") or "baseline"), status="none")
            with f3:
                ok = bool(last_fc.get("success"))
                c.kpi_tile("Status", "OK" if ok else "Error", status="ready" if ok else "error")
            tl = last_fc.get("timeline") or []
            if tl:
                last = tl[-1]
                st.caption(
                    f"End OH {last.get('operations_health')} \u00b7 End NPS {last.get('nps')} \u00b7 "
                    f"{last_fc.get('_timestamp', '')[:19]}"
                )
        else:
            c.empty_state("No forecast has been run yet. Use the **Forecast** page.",
                          icon="\U0001f4c9")

    # ---- Analytics ----------------------------------------------------------
    st.divider()
    c.section("Supporting analytics", "\U0001f4d1")
    from gui.analytics import dashboard as _da
    _da.render_analytics(
        st,
        models=svc.list_models(),
        health=svc.system_health(),
        last_forecast=svc.STATE.get_last_forecast() or {},
        status=svc.STATE.status(),
    )
