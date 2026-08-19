"""Reverse Optimizer view: generate operational states that drive OH+NPS targets.

This is the canonical OH+NPS reverse path. It generates new operational
states, evaluates each through the canonical PredictionService (predicting OH
and NPS together from the same generated state), and exposes the ranked
generated candidates. It never scans existing trained models and never calls
TargetStateEngine for this reverse OH/NPS path.
"""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import contracts as ct
from gui import services as svc

# Max candidates the canonical optimizer exposes (mirrors
# core.forecast_ai.optimization.optimizer.MAX_EXPOSED_CANDIDATES).
MAX_EXPOSED_CANDIDATES = 7

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

# session_state keys for the reverse result and the exact configuration that
# produced it. The result is only ever rendered when its stored signature
# exactly matches the current objective configuration.
REVERSE_RESULT_KEY = "reverse_result"
REVERSE_RESULT_SIGNATURE_KEY = "reverse_result_signature"


def objective_signature(
    family,
    optimise_oh,
    target_oh,
    optimise_nps,
    target_nps,
):
    """Deterministic key uniquely describing the active optimization config.

    An inactive objective contributes ``None`` for its target, so a stale
    target value left in a disabled input can never leak into the signature
    and keep an old result alive under a changed objective selection.
    """
    return (
        family,
        bool(optimise_oh),
        float(target_oh) if optimise_oh else None,
        bool(optimise_nps),
        float(target_nps) if optimise_nps else None,
    )


def active_targets(optimise_oh, target_oh, optimise_nps, target_nps):
    """Return ONLY the targets for the ACTIVE objectives.

    ``reverse_optimize_canonical`` receives exactly these keyword targets; an
    inactive objective is omitted so its stale target value is never forwarded
    to the canonical ReverseOptimizer.
    """
    targets = {}
    if optimise_oh:
        targets["target_oh"] = float(target_oh)
    if optimise_nps:
        targets["target_nps"] = float(target_nps)
    return targets


def store_reverse_result(session, result, signature):
    """Persist a result together with the exact config that produced it."""
    session[REVERSE_RESULT_KEY] = result
    session[REVERSE_RESULT_SIGNATURE_KEY] = signature


def current_reverse_result(session, signature):
    """Return the stored result ONLY when its signature matches the current
    configuration.

    If the current config differs (objective/target/family changed) the stored
    result is stale: it is dropped and ``None`` is returned so the GUI shows
    the current objective form and never renders an old result under a new
    configuration.
    """
    stored_signature = session.get(REVERSE_RESULT_SIGNATURE_KEY)
    result = session.get(REVERSE_RESULT_KEY)
    if result is not None and stored_signature == signature:
        return result
    if result is not None:
        session.pop(REVERSE_RESULT_KEY, None)
        session.pop(REVERSE_RESULT_SIGNATURE_KEY, None)
    return None


def render() -> None:
    c.page_title("Reverse Optimizer", eyebrow="Optimization",
                 help_text="Reverse-optimise KPIs for OH & NPS targets")

    from gui import model_selection as ms

    c.section("Model", "🛰️")
    option = ms.render_model_selector(feature="reverse")
    family = option.family if option is not None else None

    st.markdown(
        "Set an **Operational Health** and/or **NPS** target. The engine "
        "generates new operational states and evaluates each through the "
        "canonical PredictionService — OH and NPS are predicted together from "
        "the same generated state, and multiple ranked candidates are shown."
    )

    c.section("Objective", "🎯")
    # Objective widgets live OUTSIDE a form so their state reflects the current
    # selection on every rerun (the authoritative source for the config
    # signature). Only the Run action is a button.
    col_oh, col_nps = st.columns(2)
    with col_oh:
        optimise_oh = st.checkbox("Optimise OH", value=True, key="rev_optimise_oh")
        target_oh = st.number_input(
            "Target OH",
            ct.OH_MIN, ct.OH_MAX,
            ct.kpi_default("operations_health"),
            step=1.0,
            disabled=not optimise_oh,
            key="rev_target_oh",
            help=f"Desired Operational Health value ({ct.OH_MIN:g}–{ct.OH_MAX:g}).",
        )
    with col_nps:
        optimise_nps = st.checkbox("Optimise NPS", value=False, key="rev_optimise_nps")
        target_nps = st.number_input(
            "Target NPS",
            ct.NPS_MIN, ct.NPS_MAX,
            ct.kpi_default("nps"),
            step=1.0,
            disabled=not optimise_nps,
            key="rev_target_nps",
            help=f"Desired NPS value ({ct.NPS_MIN:g}–{ct.NPS_MAX:g}).",
        )

    at_least_one = bool(optimise_oh) or bool(optimise_nps)
    if not at_least_one:
        st.warning("Select at least one objective: OH or NPS.")

    run = st.button(
        "▶  Run Reverse Optimization",
        type="primary",
        disabled=(option is None) or not at_least_one,
        width="stretch",
        key="rev_run",
    )

    signature = objective_signature(
        family, optimise_oh, target_oh, optimise_nps, target_nps,
    )

    if run:
        if not at_least_one:
            # Button is disabled when no objective is selected; guard anyway so
            # the optimizer is never invoked with an empty objective.
            st.warning("Select at least one objective: OH or NPS.")
        else:
            with st.spinner("Generating and evaluating candidate operational states…"):
                result = c.guarded(
                    svc.reverse_optimize_canonical,
                    family=family,
                    **active_targets(optimise_oh, target_oh, optimise_nps, target_nps),
                )
            if result is not None:
                store_reverse_result(st.session_state, result, signature)
            else:
                # guarded() already surfaced the exception as an st.error; make
                # sure we never silently keep a stale result underneath it.
                st.session_state.pop(REVERSE_RESULT_KEY, None)
                st.session_state.pop(REVERSE_RESULT_SIGNATURE_KEY, None)

    # Render the stored result ONLY when its configuration signature exactly
    # matches the current one. Any objective/target/family change invalidates it
    # immediately on the next rerun; an unchanged config keeps it visible across
    # internal reruns.
    result = current_reverse_result(st.session_state, signature)
    if not result:
        if not at_least_one:
            c.empty_state(
                "Select at least one objective: OH or NPS, then run the optimizer.",
                icon="🎯",
            )
        else:
            c.empty_state("Set an objective above and run the optimizer to see "
                          "recommended KPIs.", icon="🔄")
        return

    st.divider()
    ms.render_result_model(result.get("active_family"), option)
    _render_result(result)


def _fmt(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _render_result(result) -> None:
    # Never silently swallow an optimizer error.
    errors = result.get("errors") or []
    for err in errors:
        st.error(str(err))

    if result.get("abstained"):
        st.warning("No target supplied. Provide target OH and/or target NPS.")
        c.raw_json_expander(result)
        return

    st.markdown("#### REVERSE OPTIMIZATION RESULT")

    status = result.get("status")
    if result.get("success"):
        st.success(status or "Target reached within tolerance.")
    else:
        st.warning(status or "Target not reached — showing closest generated state.")

    # Joint target / predicted metrics. OH and NPS are always predicted from
    # the same generated state, even when only one target is supplied.
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        c.kpi_tile("Target OH", _fmt(result.get("target_oh")), status="none")
    with m2:
        c.kpi_tile("Predicted OH", _fmt(result.get("predicted_oh")),
                   status="ready" if result.get("predicted_oh") is not None else "none")
    with m3:
        c.kpi_tile("Target NPS", _fmt(result.get("target_nps")), status="none")
    with m4:
        c.kpi_tile("Predicted NPS", _fmt(result.get("predicted_nps")),
                   status="ready" if result.get("predicted_nps") is not None else "none")

    d1, d2 = st.columns(2)
    with d1:
        c.kpi_tile("Distance / target gap",
                   f"{result.get('distance'):.3f}" if result.get("distance") is not None else "—",
                   status="none")
    with d2:
        c.kpi_tile("Achieved",
                   "Yes" if result.get("success") else "No",
                   status="ready" if result.get("success") else "none")

    # Recommended operational state (changed KPIs + state variables).
    rec = result.get("recommended_state") or {}
    changes = result.get("state_changes") or {}
    if rec:
        c.section("Recommended operational state", "🛠️")
        rows = []
        for key, label in STATE_LABELS.items():
            if key in rec:
                rows.append({
                    "KPI": label,
                    "Value": rec[key],
                    "Change": changes.get(key),
                })
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            c.raw_json_expander(rec, label="Recommended state (raw)")

    # Ranked generated candidates.
    candidates = result.get("candidates") or []
    if candidates:
        c.section(f"Generated candidates ({len(candidates)})", "🧮")
        for cand in candidates[:MAX_EXPOSED_CANDIDATES]:
            _render_candidate(cand)

    st.caption("Optimization basis: joint OH+NPS from the same generated state.")
    c.raw_json_expander(result)


def _render_candidate(cand) -> None:
    rank = cand.get("rank")
    name = cand.get("name") or f"Candidate {rank}"
    feasible = bool(cand.get("feasible"))
    title = f"{rank}. {name} — {'Feasible / target achieved' if feasible else 'Target not achieved'}"
    with st.expander(title, expanded=rank == 1):
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            c.kpi_tile("Predicted OH", _fmt(cand.get("predicted_operations_health")),
                       status="ready" if cand.get("predicted_operations_health") is not None else "none")
        with cc2:
            c.kpi_tile("Predicted NPS", _fmt(cand.get("predicted_nps")),
                       status="ready" if cand.get("predicted_nps") is not None else "none")
        with cc3:
            c.kpi_tile("OH error", _fmt(cand.get("operations_health_error")), status="none")
        with cc4:
            c.kpi_tile("NPS error", _fmt(cand.get("nps_error")), status="none")

        ci = cand.get("confidence_interval") or {}
        if ci.get("p05") is not None and ci.get("p95") is not None:
            st.write(
                f"**Canonical NPS 90% interval:** [{ci['p05']:.1f}, {ci['p95']:.1f}] "
                f"(median {ci.get('p50') if ci.get('p50') is not None else '—'})"
            )
            if ci.get("basis"):
                st.caption(f"Basis: {ci['basis']}")

        if cand.get("explanation"):
            st.write(f"**Explanation:** {cand['explanation']}")
        if cand.get("rank_reason"):
            st.write(f"**Why this rank:** {cand['rank_reason']}")

        changes = cand.get("key_operational_changes") or cand.get("state_changes") or {}
        if changes:
            st.write("**Key operational changes:**")
            for key, val in changes.items():
                st.write(f"- {STATE_LABELS.get(key, key)}: {val:g}")
