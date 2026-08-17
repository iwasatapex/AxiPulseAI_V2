"""Target State analytics: gap + feasibility for the inverse problem.

The target-state engine finds an operational state whose predicted outcome
hits the requested targets. Analytics explains the current-vs-desired gap
and whether the solution is feasible within canonical bounds.
"""
from __future__ import annotations

from typing import Any, Dict, List

from gui.analytics import common as a
from gui import contracts as ct

# Engine result key -> (label, contract KPI key)
_TARGET_LABELS = {
    "operational_health": ("Operational Health", "operations_health"),
    "nps": ("NPS", "nps"),
    "release": ("Release Rate", "release"),
    "transfer": ("Transfer Rate", "transfer"),
    "quality": ("Quality", "quality"),
    "competency": ("Competency", "competency"),
    "attendance": ("Attendance", "attendance"),
}
# Achieved-value key used by the engine's consensus block per target.
_CONSENSUS_KEY = {
    "operational_health": "oh",
    "nps": "nps",
    "release": "release",
    "transfer": "transfer",
    "quality": "quality",
    "competency": "competency",
    "attendance": "attendance",
}


def gap_analysis(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-target: desired, achieved, delta, direction."""
    targets = result.get("targets") or {}
    consensus = result.get("consensus") or {}
    rows: List[Dict[str, Any]] = []
    for key, desired_raw in targets.items():
        if key not in _TARGET_LABELS:
            continue
        label, _ = _TARGET_LABELS[key]
        desired = a.fnum(desired_raw)
        achieved = a.fnum(consensus.get(_CONSENSUS_KEY.get(key, key)))
        delta = None
        direction = None
        if desired is not None and achieved is not None:
            delta = achieved - desired
            direction = "over" if delta > 0 else ("under" if delta < 0 else "met")
        rows.append({
            "target": label, "desired": desired, "achieved": achieved,
            "delta": delta, "direction": direction,
        })
    return rows


def feasibility(result: Dict[str, Any]) -> Dict[str, Any]:
    """Feasible / infeasible + hard-bound conflicts + distance."""
    rec = result.get("recommended_state") or {}
    targets = result.get("targets") or {}
    conflicts = []
    for key, _ in targets.items():
        if key not in _TARGET_LABELS:
            continue
        _, ck = _TARGET_LABELS[key]
        cfg = ct.KPI.get(ck)
        if not cfg or cfg.get("target") is None:
            continue
        val = a.fnum(rec.get(key) if key in rec else None)
        if val is not None and not (cfg["min"] <= val <= cfg["max"]):
            conflicts.append(
                f"{_TARGET_LABELS[key][0]} value {val} outside [{cfg['min']:g},{cfg['max']:g}]"
            )
    distance = a.fnum(result.get("distance"))
    found = bool(rec)
    if conflicts or not found:
        feasible = False
    elif distance is not None and distance > 0.15:
        feasible = False
    else:
        feasible = True
    return {
        "feasible": feasible,
        "found_solution": found,
        "distance": distance,
        "conflicts": conflicts,
        "note": "Feasibility derived from canonical hard bounds, solution existence, and "
                "distance (threshold 0.15, documented).",
    }


def optimization_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:
    boards = result.get("leaderboards") or {}
    return {
        "distance": a.fnum(result.get("distance")),
        "found_solution": bool(result.get("recommended_state")),
        "oh_models": len(boards.get("OH", []) or []) if isinstance(boards, dict) else 0,
        "nps_models": len(boards.get("NPS", []) or []) if isinstance(boards, dict) else 0,
    }


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def render_analytics(st, result: Dict[str, Any]) -> None:
    import pandas as pd
    import plotly.graph_objects as go

    st.markdown("## Analytics")
    st.caption("Target-state analytics explain the inverse problem the "
               "TargetStateEngine solved.")

    with st.expander("Desired vs Achieved (Gap)", expanded=True):
        rows = gap_analysis(result)
        if rows:
            data = []
            for r in rows:
                data.append({
                    # Display-only copies normalized to strings (Arrow-safe).
                    "Target": a.disp(r["target"]),
                    "Desired": a.disp(r["desired"]),
                    "Achieved": a.disp(r["achieved"]),
                    "Delta": a.disp(r["delta"]),
                    "Direction": r["direction"] or "—",
                })
            st.dataframe(data, width="stretch", hide_index=True)
            fig = go.Figure()
            names = [r["target"] for r in rows]
            desired = [r["desired"] or 0 for r in rows]
            achieved = [r["achieved"] or 0 for r in rows]
            fig.add_trace(go.Bar(name="Desired", x=names, y=desired, marker_color="#3b82f6"))
            fig.add_trace(go.Bar(name="Achieved", x=names, y=achieved, marker_color="#22c55e"))
            fig.update_layout(barmode="group", title=dict(text="Achieved vs desired target"),
                              height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch", key="analytics_target_state_achieved_vs_desired")
        else:
            st.info(a.NA)

    with st.expander("Feasibility", expanded=True):
        f = feasibility(result)
        st.metric("Feasible", "Yes" if f["feasible"] else "No")
        if f["conflicts"]:
            for c in f["conflicts"]:
                st.error(c)
        st.write("Found solution:", f["found_solution"])
        st.write("Distance:", f["distance"])
        st.caption(f["note"])

    with st.expander("Optimization Diagnostics", expanded=False):
        d = optimization_diagnostics(result)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Distance", f"{d['distance']:.3f}" if d["distance"] is not None else "—")
        m2.metric("Solution found", "Yes" if d["found_solution"] else "No")
        m3.metric("OH council models", d["oh_models"])
        m4.metric("NPS council models", d["nps_models"])
        st.caption("Iterations/convergence/objective are not exposed by the target-state engine.")
