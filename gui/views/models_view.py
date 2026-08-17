"""Models view: list complete model families and select the active family."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from gui import components as c
from gui import services as svc


def _fmt_mae(value) -> str:
    """Format an MAE value as human-readable, never a raw dict."""
    if value is None:
        return "—"
    if isinstance(value, dict):
        vals = [v for v in value.values() if isinstance(v, (int, float))]
        if not vals:
            return "—"
        value = min(vals)  # diagnostic: best (lowest) error across algorithms
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_selected_mae(value, algorithm) -> str:
    """Format the MAE for the *selected* algorithm, never a cross-algorithm min.

    When ``value`` is a per-algorithm dict, return the entry belonging to
    ``algorithm`` (the model actually selected for this row).  It must not
    silently substitute ``min(all_algorithm_metrics)`` as the selected model's
    metric — that would attribute another algorithm's result to this one.
    """
    if value is None:
        return "—"
    if isinstance(value, dict):
        # Selected algorithm's own metric, not the best-of-all-algorithms.
        own = value.get(algorithm)
        if own is not None:
            return _fmt_mae(own)
        # Fall back to a human-readable line listing each algorithm's metric
        # so no single foreign number is silently attributed to the selection.
        parts = []
        for alg, err in sorted(value.items()):
            if isinstance(err, (int, float)):
                parts.append(f"{alg}: {err:.2f}")
        return " · ".join(parts) if parts else "—"
    return _fmt_mae(value)


def _fmt_int(value) -> str:
    """Format an integer count with thousands separators."""
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _diagnostics_for(model_info, algorithm):
    """Return the model_selection_diagnostics entry for ``algorithm``."""
    diags = model_info.get("model_selection_diagnostics") or {}
    if isinstance(diags, dict):
        return diags.get(algorithm) or diags.get(model_info.get("model_name"))
    return {}


def _feasible_label(model_info, algorithm) -> str:
    d = _diagnostics_for(model_info, algorithm)
    if d and isinstance(d, dict) and "final_fit_feasible" in d:
        return "Yes" if d["final_fit_feasible"] else "No"
    return "—"


def _est_mem_label(model_info, algorithm) -> str:
    d = _diagnostics_for(model_info, algorithm)
    mem = d.get("final_fit_estimated_memory_mb") if isinstance(d, dict) else None
    if mem is None:
        return "—"
    try:
        return f"{float(mem):,.0f} MB"
    except (TypeError, ValueError):
        return "—"


def _excl_reason_label(model_info, algorithm) -> str:
    d = _diagnostics_for(model_info, algorithm)
    if not isinstance(d, dict):
        return "—"
    reason = d.get("reason_if_not_feasible")
    return reason if reason else "—"


def render() -> None:
    c.page_title("Models", help_text="Complete OH+NPS model families")

    models = svc.list_models()
    active = svc.STATE.get_active_family()

    if not models:
        c.empty_state(
            "No complete model pairs found. "
            "A family needs both `{name}_OH.pkl` and `{name}_NPS.pkl`.",
            icon="🗂️",
        )
        return

    # ---- Active selection ----
    st.markdown("#### Select Active Model Family")
    st.caption("Only families with **both** OH and NPS files are shown. "
               "Selection is explicit — never silent.")
    from gui import model_selection as ms
    include_test = st.checkbox(
        "Show test/staging models",
        value=bool(st.session_state.get("apgui_models_include_test", False)),
        key="apgui_models_include_test",
        help="Smoke/test/staging model pairs are hidden from selection unless "
             "this is enabled.",
    )
    options = [
        m["family"] for m in models
        if "error" not in m and (include_test or not ms.is_test_family(m["family"]))
    ]
    if not options:
        c.empty_state("No selectable model families. Enable test/staging models "
                      "above or train one on the Train page.", icon="⚠️")
        return
    idx = options.index(active) if active in options else 0
    chosen = st.selectbox("Model family", options=options, index=idx)
    if chosen != active:
        try:
            svc.select_model_family(chosen)
            st.success(f"Active model family set to **{chosen}**")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not activate {chosen}: {exc}")

    # ---- Table (leaderboard) ----
    rows = []
    for m in models:
        if "error" in m:
            rows.append({"Model Family": m["family"], "Status": f"Error: {m['error']}"})
            continue
        oh = m.get("oh") or {}
        nps = m.get("nps") or {}
        oh_alg = oh.get("model_name") or oh.get("algorithm") or "—"
        nps_alg = nps.get("model_name") or nps.get("algorithm") or "—"
        rows.append({
            "Model Family": m["family"],
            "Predictor": "Operational Health",
            "Algorithm": oh_alg,
            "CV MAE": _fmt_selected_mae(oh.get("mae"), oh_alg),
            "Final Fit Feasible": _feasible_label(oh, oh_alg),
            "Estimated Memory": _est_mem_label(oh, oh_alg),
            "Exclusion Reason": _excl_reason_label(oh, oh_alg),
            "Features": oh.get("feature_count"),
            "Training Rows": _fmt_int(oh.get("training_rows")),
            "History Days": _fmt_int(oh.get("history_days")),
            "Status": "Ready" if oh.get("trained") else "—",
            "Device": oh.get("device") or "—",
        })
        rows.append({
            "Model Family": m["family"],
            "Predictor": "NPS",
            "Algorithm": nps_alg,
            "CV NPS MAE": _fmt_selected_mae(nps.get("mae"), nps_alg),
            "Final Fit Feasible": _feasible_label(nps, nps_alg),
            "Estimated Memory": _est_mem_label(nps, nps_alg),
            "Exclusion Reason": _excl_reason_label(nps, nps_alg),
            "Features": nps.get("feature_count"),
            "Training Rows": _fmt_int(nps.get("training_rows")),
            "History Days": _fmt_int(nps.get("history_days")),
            "Status": "Ready" if nps.get("trained") else "—",
            "Device": nps.get("device") or "—",
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("Leaderboard shows user-facing model information only; raw "
               "metadata stays in the Family Details technical expander.")

    # ---- Detail cards ----
    st.markdown("#### Family Details")
    for m in models:
        if "error" in m:
            continue
        with st.expander(f"{m['family']}  {'(active)' if m.get('active') else ''}", expanded=m.get("active")):
            _render_family(m)


def _render_family(m: dict) -> None:
    oh = m.get("oh") or {}
    nps = m.get("nps") or {}
    oh_alg = oh.get("model_name") or oh.get("algorithm")
    nps_alg = nps.get("model_name") or nps.get("algorithm")
    col1, col2 = st.columns(2)

    col1.markdown("**Operational Health**")
    col1.write(f"Algorithm: {oh.get('model_name') or '—'}")
    col1.write(f"Features: {oh.get('feature_count')}")
    col1.write(f"Engine: {oh.get('engine_version')}")
    col1.write(f"Status: {'Ready' if oh.get('trained') else '—'}")
    col1.write(f"CV MAE: {_fmt_selected_mae(oh.get('mae'), oh_alg)}")
    col1.write(f"Final Fit Feasible: {_feasible_label(oh, oh_alg)}")
    col1.write(f"Estimated Memory: {_est_mem_label(oh, oh_alg)}")
    col1.write(f"Exclusion Reason: {_excl_reason_label(oh, oh_alg)}")
    col1.write(f"History Days: {_fmt_int(oh.get('history_days'))}")
    col1.write(f"Device: {oh.get('device') or '—'}")
    _render_algo_perf(col1, oh.get("algorithm_performance"))

    col2.markdown("**NPS**")
    col2.write(f"Algorithm: {nps.get('model_name') or '—'}")
    col2.write(f"Features: {nps.get('feature_count')}")
    col2.write(f"Engine: {nps.get('engine_version')}")
    col2.write(f"Status: {'Ready' if nps.get('trained') else '—'}")
    col2.write(f"NPS MAE: {_fmt_selected_mae(nps.get('mae'), nps_alg)}")
    col2.write(f"History Days: {_fmt_int(nps.get('history_days'))}")
    col2.write(f"Device: {nps.get('device') or '—'}")
    _render_algo_perf(col2, nps.get("algorithm_performance"))

    st.caption(f"Saved: {m.get('saved_at', '')[:19]}")
    c.raw_json_expander({"family": m["family"], "oh": oh, "nps": nps})


def _render_algo_perf(col, perf) -> None:
    """Render per-algorithm error metrics as human-readable lines, never raw dicts."""
    if not perf:
        return
    col.caption("Per-algorithm error (best = lowest):")
    if isinstance(perf, dict):
        for alg, err in sorted(perf.items(), key=lambda kv: (kv[1] if isinstance(kv[1], (int, float)) else float("inf"))):
            col.write(f"- {alg}: {_fmt_mae(err)}")
    else:
        col.write(f"- {perf}")
