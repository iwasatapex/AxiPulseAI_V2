"""Dashboard / system analytics: model inventory, health, KPI overview,
recent activity. Consumes session state and system_health output.
"""
from __future__ import annotations

from typing import Any, Dict, List

from gui.analytics import common as a
from gui import contracts as ct


def model_inventory(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for m in models:
        if "error" in m:
            rows.append({"family": m["family"], "status": "Error", "oh_model": None,
                         "nps_model": None, "active": False})
            continue
        rows.append({
            "family": m["family"],
            "status": "Ready",
            "oh_model": (m.get("oh") or {}).get("model_name"),
            "nps_model": (m.get("nps") or {}).get("model_name"),
            "active": bool(m.get("active")),
            "saved_at": (m.get("saved_at") or "")[:19],
        })
    return rows


def health_breakdown(health: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = health.get("checks") or {}
    out = []
    for key in ("models", "active_model", "scenarios"):
        chk = checks.get(key) or {}
        out.append({
            "component": key,
            "status": chk.get("status", "Unknown"),
            "detail": chk.get("detail") or chk.get("family") or (", ".join(chk.get("available_families") or [])),
        })
    return out


def kpi_overview(last_forecast: Dict[str, Any]) -> List[Dict[str, Any]]:
    """KPI target overview from the latest forecast's final day (if any)."""
    timeline = (last_forecast or {}).get("timeline") or []
    if not timeline:
        return []
    final = timeline[-1]
    rows = []
    for key in ["quality", "competency", "attendance", "release", "transfer"]:
        if key in final:
            val = a.fnum(final.get(key))
            target = ct.kpi_target(key)
            gap = (target - val) if val is not None and target is not None else None
            rows.append({
                "kpi": ct.KPI[key]["label"],
                "latest": val,
                "target": target,
                "gap": gap,
                "met": ct.kpi_met(key, val),
            })
    return rows


def recent_activity(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = [
        ("Prediction", status.get("last_prediction_at")),
        ("Forecast", status.get("last_forecast_at")),
        ("ADIE decision", status.get("last_adie_at")),
    ]
    return [{"activity": label, "time": ts or "—"} for label, ts in items]


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def render_analytics(st, models: List[Dict[str, Any]], health: Dict[str, Any],
                     last_forecast: Dict[str, Any], status: Dict[str, Any]) -> None:
    import pandas as pd

    st.markdown("## Analytics")
    st.caption("System-wide operational overview derived from session state "
               "and the readiness check. No model is run to render this.")

    with st.expander("Model Inventory", expanded=True):
        inv = model_inventory(models)
        if inv:
            data = [{
                "Family": r["family"], "Status": r["status"],
                "OH model": r["oh_model"], "NPS model": r["nps_model"],
                "Active": "★" if r["active"] else "",
            } for r in inv]
            st.dataframe(data, width="stretch", hide_index=True)
        else:
            st.info("No model families available.")

    with st.expander("System Health", expanded=True):
        m1, m2 = st.columns(2)
        m1.metric("Overall", health.get("status"))
        m2.metric("Available families", len(health.get("available_families") or []))
        hb = health_breakdown(health)
        st.dataframe(pd.DataFrame(hb), width="stretch", hide_index=True)

    with st.expander("KPI Overview (latest forecast)", expanded=False):
        rows = kpi_overview(last_forecast)
        if rows:
            data = [{
                # Display-only copies: every column is normalized to a string so
                # Streamlit's Arrow serializer never sees a mixed-type column.
                "KPI": r["kpi"],
                "Latest": a.disp(r["latest"]),
                "Target": a.disp(r["target"]),
                "Gap": a.disp(r["gap"]),
                "Met": "Yes" if r["met"] is True else ("No" if r["met"] is False else "—"),
            } for r in rows]
            st.dataframe(data, width="stretch", hide_index=True)
            st.caption("Transfer: lower is better.")
        else:
            st.info("Run a forecast to populate the KPI overview.")

    with st.expander("Recent Activity", expanded=False):
        act = recent_activity(status)
        st.dataframe(pd.DataFrame(act), width="stretch", hide_index=True)
