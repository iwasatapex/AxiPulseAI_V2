"""Reverse Optimizer analytics: required operational state + feasibility.

Pure functions consume the ``reverse_optimize`` result. No optimizer
mathematics is re-run here.
"""
from __future__ import annotations

from typing import Any, Dict, List

from gui.analytics import common as a
from gui import contracts as ct

_LABELS = {
    "quality": "Quality",
    "competency": "Competency",
    "attendance": "Attendance",
    "release": "Release Rate",
    "transfer": "Transfer Rate",
    "total_calls_received": "Total Calls Received",
    "operational_health": "Operational Health",
    "nps": "NPS",
}

# Feasibility classification thresholds (documented, conservative).
_MARGINAL_DISTANCE = 0.10
_INFEASIBLE_DISTANCE = 0.25


def required_state(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The required KPI values the optimizer recommends."""
    rec = result.get("recommended_state") or {}
    rows = []
    for key, label in _LABELS.items():
        if key in rec:
            val = a.fnum(rec[key])
            cfg = ct.KPI.get(key)
            rows.append({
                "kpi": label, "value": val, "unit": cfg.get("unit", "") if cfg else "",
                "lo": cfg["min"] if cfg else None, "hi": cfg["max"] if cfg else None,
                "near_limit": val is not None and cfg and (val == cfg["min"] or val == cfg["max"]),
            })
    return rows


def target_vs_predicted(result: Dict[str, Any]) -> Dict[str, Any]:
    target = a.fnum(result.get("target"))
    predicted = a.fnum(result.get("predicted"))
    return {
        "metric": result.get("metric"),
        "target": target,
        "predicted": predicted,
        "delta": (predicted - target) if target is not None and predicted is not None else None,
    }


def constraint_analysis(result: Dict[str, Any]) -> List[str]:
    """Flag recommended values sitting on canonical hard limits."""
    flags = []
    for r in required_state(result):
        if r["near_limit"]:
            flags.append(f"{r['kpi']} is at the canonical limit {r['value']}.")
    return flags


def optimization_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "distance": a.fnum(result.get("distance")),
        "found": bool(result.get("found")),
        "convergence": a.safe_get(result, "optimizer", "converged"),
        "iterations": a.safe_get(result, "optimizer", "iterations"),
    }


def feasibility_classification(result: Dict[str, Any]) -> Dict[str, Any]:
    """Classify Feasible / Marginal / Infeasible (thresholds documented)."""
    distance = a.fnum(result.get("distance"))
    found = bool(result.get("found")) and bool(result.get("recommended_state"))
    if not found:
        cls = "Infeasible"
    elif distance is None:
        cls = "Marginal"
    elif distance <= _MARGINAL_DISTANCE:
        cls = "Feasible"
    elif distance <= _INFEASIBLE_DISTANCE:
        cls = "Marginal"
    else:
        cls = "Infeasible"
    return {
        "class": cls,
        "distance": distance,
        "note": f"Feasible ≤{_MARGINAL_DISTANCE} · Marginal ≤{_INFEASIBLE_DISTANCE} · else Infeasible.",
    }


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def render_analytics(st, result: Dict[str, Any]) -> None:
    import pandas as pd

    st.markdown("## Analytics")
    st.caption("Reverse-optimizer analytics explain the inverse relationship "
               "between the desired output and the required operational state.")

    with st.expander("Desired Output vs Required State", expanded=True):
        tvp = target_vs_predicted(result)
        m1, m2, m3 = st.columns(3)
        m1.metric("Target", f"{tvp['target']:.1f}" if tvp["target"] is not None else "—")
        m2.metric("Predicted achieved", f"{tvp['predicted']:.1f}" if tvp["predicted"] is not None else "—")
        m3.metric("Delta", f"{tvp['delta']:.1f}" if tvp["delta"] is not None else "—")
        rows = required_state(result)
        if rows:
            data = [{
                "KPI": r["kpi"], "Required": r["value"], "Unit": r["unit"],
                "Range": f"[{r['lo']:g}, {r['hi']:g}]" if r["lo"] is not None else "—",
            } for r in rows]
            st.dataframe(data, width="stretch", hide_index=True)

    with st.expander("Constraint Analysis", expanded=False):
        flags = constraint_analysis(result)
        if flags:
            for f in flags:
                st.warning(f)
        else:
            st.success("No recommended value sits on a canonical hard limit.")

    with st.expander("Optimization Quality", expanded=False):
        q = optimization_quality(result)
        st.metric("Distance", f"{q['distance']:.3f}" if q["distance"] is not None else "—")
        st.write("Solution found:", q["found"])
        if q["convergence"] is not None:
            st.write("Converged:", q["convergence"])
        if q["iterations"] is not None:
            st.write("Iterations:", q["iterations"])
        else:
            st.info("Convergence/iterations not exposed by the optimizer.")

    with st.expander("Feasibility Classification", expanded=True):
        fc = feasibility_classification(result)
        st.metric("Classification", fc["class"])
        st.caption(fc["note"])
        st.caption("A returned number alone does not make a target feasible; "
                   "classification uses solution existence + residual distance.")
