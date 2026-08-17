"""Forecast analytics: KPI trajectories, target attainment, trends, risk.

Pure functions consume the ``forecast`` payload (timeline of days + summary).
No recursive forecast is re-run here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gui.analytics import common as a
from gui import contracts as ct

# KPIs to track + the canonical direction each must move to be "better".
TRACKED = ["operations_health", "quality", "competency", "attendance", "release", "transfer", "nps"]

KPI_AXIS_RANGES = {
    "operations_health": (ct.OH_MIN, ct.OH_MAX),
    "quality": ct.kpi_bounds("quality"),
    "competency": ct.kpi_bounds("competency"),
    "attendance": ct.kpi_bounds("attendance"),
    "release": ct.kpi_bounds("release"),
    "transfer": ct.kpi_bounds("transfer"),
    "nps": (ct.NPS_MIN, ct.NPS_MAX),
}


def horizon_summary(fc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "horizon": fc.get("horizon"),
        "scenario": fc.get("scenario"),
        "family": fc.get("active_family"),
        "timestamp": (fc.get("_timestamp") or "")[:19],
        "days": len(fc.get("timeline") or []),
    }


def kpi_trajectories(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-KPI trajectory stats across the timeline."""
    out: Dict[str, Any] = {}
    for key in TRACKED:
        series = [a.fnum(d.get(key)) for d in timeline]
        series = [v for v in series if v is not None]
        if not series:
            out[key] = {"available": False}
            continue
        baseline = series[0]
        final = series[-1]
        out[key] = {
            "available": True,
            "baseline": baseline,
            "final": final,
            "min": min(series),
            "max": max(series),
            "mean": sum(series) / len(series),
            "change": final - baseline,
            "direction": "up" if final > baseline else ("down" if final < baseline else "flat"),
            "range": KPI_AXIS_RANGES.get(key, (None, None)),
        }
    return out


def target_attainment(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-day KPI-met + overall attainment across the horizon."""
    per_day = []
    for i, day in enumerate(timeline):
        met = a.day_kpi_met(day)
        per_day.append({"day": i, "met": met})
    met_days = [d for d in per_day if d["met"]]
    overall = (len(met_days) / len(per_day)) * 100.0 if per_day else None
    # First day target lost / recovered (met at day0 then lost, or lost then recovered).
    first_loss = next((d["day"] for d in per_day if not d["met"]), None)
    recovered = None
    if first_loss is not None:
        for d in per_day:
            if d["day"] > first_loss and d["met"]:
                recovered = d["day"]
                break
    return {
        "per_day": per_day,
        "met_days": len(met_days),
        "total_days": len(per_day),
        "pct_horizon_met": overall,
        "first_loss_day": first_loss,
        "first_recovery_day": recovered,
    }


def trend_analytics(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Day-over-day deltas, rolling mean (if horizon supports it), acceleration."""
    out: Dict[str, Any] = {}
    for key in ["operations_health", "nps"]:
        series = [a.fnum(d.get(key)) for d in timeline]
        series = [v for v in series if v is not None]
        if len(series) < 2:
            out[key] = {"available": False}
            continue
        deltas = [round(series[i] - series[i - 1], 3) for i in range(1, len(series))]
        window = min(3, len(series))
        rolling = [round(sum(series[max(0, i - window + 1):i + 1]) / len(series[max(0, i - window + 1):i + 1]), 3)
                   for i in range(len(series))]
        out[key] = {
            "available": True,
            "deltas": deltas,
            "rolling_mean": rolling,
            "acceleration": round(deltas[-1] - deltas[0], 3) if len(deltas) > 1 else None,
        }
    return out


def uncertainty(fc: Dict[str, Any]) -> Dict[str, Any]:
    """Use actual forecast intervals if the engine exposed them."""
    intervals = []
    timeline = fc.get("timeline") or []
    for i, d in enumerate(timeline):
        lo = a.fnum(d.get("operations_health_lower") or d.get("lower") or d.get("low"))
        hi = a.fnum(d.get("operations_health_upper") or d.get("upper") or d.get("high"))
        if lo is not None and hi is not None:
            intervals.append({"day": i, "lower": lo, "upper": hi})
    return {
        "available": bool(intervals),
        "intervals": intervals,
        "note": "Engine-provided forecast intervals." if intervals else (
            "Forecast intervals not exposed by the engine; uncertainty is not assumed."
        ),
    }


def risk_flags(timeline: List[Dict[str, Any]]) -> List[str]:
    """Derive risk flags from the actual forecast output + canonical thresholds."""
    flags: List[str] = []
    if not timeline:
        return flags
    oh_series = [a.fnum(d.get("operations_health")) for d in timeline]
    if len(oh_series) >= 2 and all(v is not None for v in oh_series):
        if oh_series[-1] < oh_series[0]:
            flags.append("Declining operational health over the horizon.")
    last = timeline[-1]
    transfer = a.fnum(last.get("transfer"))
    if transfer is not None and transfer > ct.kpi_target("transfer"):
        flags.append(f"Transfer ({transfer:.1f}) exceeds the canonical target ({ct.kpi_target('transfer')}).")
    release = a.fnum(last.get("release"))
    if release is not None and release < ct.kpi_target("release"):
        flags.append(f"Release ({release:.1f}) below the canonical target ({ct.kpi_target('release')}).")
    nps_series = [a.fnum(d.get("nps")) for d in timeline]
    if len(nps_series) >= 2 and all(v is not None for v in nps_series):
        if nps_series[-1] < nps_series[0]:
            flags.append("NPS deterioration over the horizon.")
    return flags


def _final_of(traj: Dict[str, Any], key: str):
    t = traj.get(key)
    return t["final"] if t and t.get("available") else None


def _delta_of(traj: Dict[str, Any], baseline_traj: Dict[str, Any], key: str):
    f = _final_of(traj, key)
    b = _final_of(baseline_traj, key)
    return (f - b) if f is not None and b is not None else None


def scenario_comparison(forecasts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare multiple enabled forecast payloads.

    Deltas are computed against the ``baseline`` forecast when present.
    Each forecast must already have been produced by the engine; no scenario
    is executed here. Disabled scenarios are never passed in by the views.
    """
    out: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for f in forecasts:
        sid = f.get("scenario")
        if sid is not None:
            by_id.setdefault(sid, f)
    baseline = by_id.get(ct.BASELINE_SCENARIO_ID)
    baseline_traj = kpi_trajectories(baseline.get("timeline") or []) if baseline is not None else None
    baseline_att = target_attainment(baseline.get("timeline") or []) if baseline is not None else None

    for fc in forecasts:
        timeline = fc.get("timeline") or []
        att = target_attainment(timeline)
        traj = kpi_trajectories(timeline)
        row = {
            "scenario": fc.get("scenario"),
            "oh_final": _final_of(traj, "operations_health"),
            "nps_final": _final_of(traj, "nps"),
            "quality_final": _final_of(traj, "quality"),
            "competency_final": _final_of(traj, "competency"),
            "release_final": _final_of(traj, "release"),
            "transfer_final": _final_of(traj, "transfer"),
            "kpi_met_pct": att["pct_horizon_met"],
            "met_days": att["met_days"],
            "total_days": att["total_days"],
            "oh_delta": _delta_of(traj, baseline_traj, "operations_health") if baseline_traj else None,
            "nps_delta": _delta_of(traj, baseline_traj, "nps") if baseline_traj else None,
            "kpi_met_pct_delta": None,
        }
        if baseline_att is not None:
            b = baseline_att["pct_horizon_met"]
            c = att["pct_horizon_met"]
            if b is not None and c is not None:
                row["kpi_met_pct_delta"] = c - b
        out.append(row)
    return out


# ---------------------------------------------------------------------
# Scenario comparison execution (reuses the existing forecast service)
# ---------------------------------------------------------------------

def _enabled_scenario_ids() -> set:
    """Return the set of scenario ids that may be executed (enabled + baseline)."""
    from gui import services as _svc
    return {s["id"] for s in _svc.list_scenarios()}


def _cache_key(state: Dict[str, Any], horizon: int, family: Any, scenario: str):
    """Stable per-request cache key (no cross-user state)."""
    try:
        import json
        state_sig = json.dumps(state, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        state_sig = str(sorted(state.items()))
    return (family, horizon, scenario, state_sig)


def run_scenario_comparison(state: Dict[str, Any], horizon: int, family: Any,
                            scenarios: List[str], forecast_fn=None,
                            cache: Optional[Dict[Any, Any]] = None):
    """Run one forecast per selected scenario via the existing forecast service.

    - Disabled/unknown scenarios are recorded as errors and NEVER executed.
    - Exactly one baseline is included (as reference) if selected.
    - Duplicate scenario ids are de-duplicated.
    - ``cache`` is a per-session dict (caller-owned); cached entries are
      reused so the same scenario is not rerun unnecessarily.
    - ``forecast_fn(state, horizon, scenario, family)`` may be injected for
      tests (default: ``gui.services.forecast`` with ``update_state=False``).

    Returns ``(results, cache)`` where each result is:
      ``{"scenario", "payload"|None, "error"|None, "cached": bool}``.
    """
    cache = cache or {}
    if forecast_fn is None:
        def _default(state_, horizon_, scenario=None, family=None):
            from gui import services as _svc
            return _svc.forecast(state_, horizon_, scenario=scenario, family=family,
                                 update_state=False)
        forecast_fn = _default

    enabled = _enabled_scenario_ids()
    entries = []
    seen = set()

    def _push(sid):
        if sid in seen:
            return
        seen.add(sid)
        if sid == ct.BASELINE_SCENARIO_ID or sid in enabled:
            entries.append((sid, None))
        else:
            entries.append((sid, f"Scenario '{sid}' is disabled or unknown and cannot be executed."))

    _push(ct.BASELINE_SCENARIO_ID)  # baseline reference (exactly one)
    for sid in scenarios:
        _push(sid)

    results = []
    for sid, pre_err in entries:
        if pre_err:
            results.append({"scenario": sid, "payload": None, "error": pre_err, "cached": False})
            continue
        key = _cache_key(state, horizon, family, sid)
        if key in cache:
            results.append({"scenario": sid, "payload": cache[key], "error": None, "cached": True})
            continue
        try:
            payload = forecast_fn(state, horizon, scenario=sid, family=family)
            cache[key] = payload
            results.append({"scenario": sid, "payload": payload, "error": None, "cached": False})
        except Exception as exc:  # noqa: BLE001 - per-scenario error surfacing
            results.append({"scenario": sid, "payload": None, "error": str(exc), "cached": False})
    return results, cache


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def render_analytics(st, fc: Dict[str, Any]) -> None:
    import plotly.graph_objects as go

    timeline = fc.get("timeline") or []
    st.markdown("## Analytics")
    st.caption("Forecast analytics derived from the orchestrator's output "
               "and canonical KPI targets. No forecast is re-run.")

    with st.expander("Overview", expanded=True):
        hs = horizon_summary(fc)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Horizon", f"{hs['horizon']} days")
        m2.metric("Scenario", hs["scenario"])
        m3.metric("Model family", hs["family"])
        m4.metric("Forecast days", hs["days"])
        st.caption(f"Run at {hs['timestamp'] or '—'}")

    with st.expander("KPI Trajectories", expanded=True):
        traj = kpi_trajectories(timeline)
        if timeline:
            fig = go.Figure()
            for key, label, color in [
                ("operations_health", "Operational Health", "#22c55e"),
                ("quality", "Quality", "#3b82f6"),
                ("competency", "Competency", "#8b5cf6"),
                ("attendance", "Attendance", "#06b6d4"),
                ("release", "Release", "#f59e0b"),
                ("transfer", "Transfer", "#ef4444"),
                ("nps", "NPS", "#6366f1"),
            ]:
                t = traj.get(key)
                if not t or not t.get("available"):
                    continue
                fig.add_trace(go.Scatter(
                    x=list(range(len(timeline))), y=[a.fnum(d.get(key)) for d in timeline],
                    mode="lines+markers", name=label, line=dict(color=color, width=2),
                    hovertemplate=f"{label}: %{{y:.1f}}<extra></extra>",
                ))
            fig.update_layout(
                title=dict(text="KPI trajectories across the horizon", font=dict(size=15)),
                xaxis=dict(title="Day"), yaxis=dict(title="Value"),
                height=380, margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, width="stretch", key="analytics_forecast_kpi_trajectories")
            st.caption("Canonical ranges — OH 0-100, Quality 60-100, Competency 55-100, "
                       "Attendance 65-100, Release 50-100, Transfer 0-20, NPS -100..100. "
                       "OH and NPS are ML-predicted model outputs (not rule-based formulas).")
            rows = []
            for key, label in [("operations_health", "Operational Health"), ("quality", "Quality"),
                               ("competency", "Competency"), ("attendance", "Attendance"),
                               ("release", "Release"), ("transfer", "Transfer"), ("nps", "NPS")]:
                t = traj.get(key)
                if not t or not t.get("available"):
                    continue
                rows.append({
                    "KPI": label, "Baseline": round(t["baseline"], 2), "Final": round(t["final"], 2),
                    "Min": round(t["min"], 2), "Max": round(t["max"], 2),
                    "Mean": round(t["mean"], 2), "Change": round(t["change"], 2),
                    "Direction": t["direction"],
                })
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("No timeline returned by the forecast.")

    with st.expander("Target Attainment", expanded=False):
        at = target_attainment(timeline)
        if at["total_days"]:
            st.metric("Days KPI-met", f"{at['met_days']}/{at['total_days']}")
            if at["pct_horizon_met"] is not None:
                st.metric("% horizon meeting target", f"{at['pct_horizon_met']:.0f}%")
            st.write("First day target lost:", at["first_loss_day"] if at["first_loss_day"] is not None else "None")
            st.write("First recovery day:", at["first_recovery_day"] if at["first_recovery_day"] is not None else "None")
            met_data = pd_df(at)
            if met_data is not None and not met_data.empty:
                st.dataframe(met_data, width="stretch", hide_index=True)
            st.caption("A day is KPI-met when ≥3 of the 4 checked KPIs "
                       "(quality, competency, release, transfer) meet their canonical target.")

    with st.expander("Trend Analytics", expanded=False):
        tr = trend_analytics(timeline)
        for key, label in (("operations_health", "Operational Health"), ("nps", "NPS")):
            t = tr.get(key)
            if not t or not t.get("available"):
                st.info(f"{label}: trend not available.")
                continue
            st.markdown(f"**{label}** day-over-day deltas: {t['deltas']}")
            st.markdown(f"Rolling mean: {t['rolling_mean']}")
            if t["acceleration"] is not None:
                st.markdown(f"Acceleration (last−first delta): {t['acceleration']:.3f}")
        st.caption("Deltas/rolling means are descriptive only — no statistical significance is claimed.")

    with st.expander("Uncertainty", expanded=False):
        unc = uncertainty(fc)
        if unc["available"]:
            st.write(unc["note"])
            st.json(unc["intervals"])
        else:
            st.info(unc["note"])

    with st.expander("Risk Flags", expanded=True):
        flags = risk_flags(timeline)
        if flags:
            for f in flags:
                st.warning(f)
        else:
            st.success("No risk flags derived from the forecast output.")

    # ---- Scenario comparison (reuses the existing forecast service) ----
    with st.expander("Scenario Comparison", expanded=False):
        render_scenario_comparison(st, fc)


def pd_df(at: Dict[str, Any]):
    """Build a small DataFrame for per-day KPI-met results."""
    import pandas as pd
    return pd.DataFrame([{"Day": d["day"], "KPI-met": "Yes" if d["met"] else "No"}
                         for d in at["per_day"]])


def render_scenario_comparison(st, fc: Dict[str, Any]) -> None:
    """Scenario comparison UI in the Forecast Analytics section.

    Lets the user multi-select enabled scenarios and run them through the
    existing forecast service, reusing the current forecast's start state,
    horizon and model family. Results + cache are session-scoped.
    """
    from gui import services as svc

    family = fc.get("active_family")
    horizon = fc.get("horizon")
    state = dict((st.session_state.get("forecast_state") or {}) if hasattr(st, "session_state") else {})

    if not state or not family or not horizon:
        st.info("Run a forecast first to populate the scenario comparison inputs.")
        return

    options = [s["id"] for s in svc.list_scenarios()]
    non_baseline = [s for s in options if s != ct.BASELINE_SCENARIO_ID]
    if not non_baseline:
        st.info("No additional enabled scenarios are available for comparison.")
        return
    default = [ct.BASELINE_SCENARIO_ID, non_baseline[0]]
    chosen = st.multiselect("Scenarios to compare", options=options, default=default,
                            help="Baseline is always included as the reference. "
                                 "Only enabled scenarios can be selected.")
    if not chosen:
        st.info("Select at least one scenario to compare.")
        return

    if st.button("Run Scenario Comparison", type="secondary"):
        cache = dict(st.session_state.get("_fc_scenario_cache") or {})
        # Seed the cache with the current single-scenario result when its
        # scenario is part of the comparison, so we never rerun it.
        cur_sid = fc.get("scenario")
        if fc.get("success") and cur_sid:
            cache.setdefault(_cache_key(state, horizon, family, cur_sid), fc)
        results, cache = run_scenario_comparison(state, horizon, family, chosen, cache=cache)
        st.session_state["_fc_scenario_cache"] = cache
        st.session_state["_fc_scenario_results"] = results

    results = st.session_state.get("_fc_scenario_results")
    if results:
        _render_comparison(st, results)


def _render_comparison(st, results) -> None:
    """Render the compact scenario-comparison summary + trajectory charts."""
    import pandas as pd
    import plotly.graph_objects as go

    ok = [r for r in results if r.get("payload")]
    errs = [r for r in results if r.get("error")]

    # 1. Summary table + delta vs baseline.
    comp = scenario_comparison([r["payload"] for r in ok])
    if comp:
        rows = []
        for c in comp:
            rows.append({
                "Scenario": c["scenario"],
                "Final OH": _r1(c["oh_final"]),
                "Final NPS": _r1(c["nps_final"]),
                "Final Quality": _r1(c["quality_final"]),
                "Final Competency": _r1(c["competency_final"]),
                "Final Release": _r1(c["release_final"]),
                "Final Transfer": _r1(c["transfer_final"]),
                "KPI-met %": _r1(c["kpi_met_pct"]),
                "OH Δ vs baseline": _r1(c["oh_delta"]),
                "NPS Δ vs baseline": _r1(c["nps_delta"]),
                "KPI-met Δ": _r1(c["kpi_met_pct_delta"]),
            })
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption("Δ = final value relative to baseline. KPI-met rule: a day meets "
                   "when ≥3 of 4 checked KPIs (quality, competency, release, transfer) "
                   "meet their canonical target. Transfer: lower is better.")

    # 2. OH trajectory.
    fig_oh = go.Figure()
    for r in ok:
        tl = r["payload"].get("timeline") or []
        fig_oh.add_trace(go.Scatter(
            x=list(range(len(tl))), y=[d.get("operations_health") for d in tl],
            mode="lines+markers", name=r["scenario"],
            hovertemplate="%{x}<br>OH: %{y:.1f}<extra></extra>",
        ))
    fig_oh.update_layout(title=dict(text="Operational Health trajectory by scenario", font=dict(size=14)),
                         xaxis=dict(title="Day"), yaxis=dict(title="OH (%)", range=[ct.OH_MIN, ct.OH_MAX]),
                         height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_oh, width="stretch", key="analytics_forecast_oh_trajectory")

    # 3. NPS trajectory (full -100..100, never clipped).
    fig_nps = go.Figure()
    for r in ok:
        tl = r["payload"].get("timeline") or []
        fig_nps.add_trace(go.Scatter(
            x=list(range(len(tl))), y=[d.get("nps") for d in tl],
            mode="lines+markers", name=r["scenario"],
            hovertemplate="%{x}<br>NPS: %{y:.1f}<extra></extra>",
        ))
    fig_nps.update_layout(title=dict(text="NPS trajectory by scenario", font=dict(size=14)),
                          xaxis=dict(title="Day"), yaxis=dict(title="NPS", range=[ct.NPS_MIN, ct.NPS_MAX]),
                          height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_nps, width="stretch", key="analytics_forecast_nps_trajectory")
    st.caption("NPS is −100..+100; negative values are shown in full. OH does not directly "
               "generate NPS — surveys produce the NPS distribution.")

    # 4. KPI-met % comparison.
    if comp:
        names = [c["scenario"] for c in comp]
        vals = [c["kpi_met_pct"] if c["kpi_met_pct"] is not None else 0 for c in comp]
        fig_k = go.Figure(go.Bar(x=names, y=vals, marker_color="#3b82f6"))
        fig_k.update_layout(title=dict(text="KPI-met % of horizon by scenario", font=dict(size=14)),
                            xaxis=dict(title="Scenario"), yaxis=dict(title="KPI-met %", range=[0, 100]),
                            height=280, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_k, width="stretch", key="analytics_forecast_kpi_met")

    # 5. Per-scenario errors (do not fail the whole comparison).
    for r in errs:
        st.error(f"Scenario '{r['scenario']}': {r['error']}")


def _r1(value):
    return round(value, 1) if isinstance(value, (int, float)) else "—"

