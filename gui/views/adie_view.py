"""ADIE Decision view: render the canonical V3 decision package with rich detail."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from gui import components as c
from gui import contracts as ct
from gui import services as svc


def _bounds(key):
    return ct.KPI[key]["min"], ct.KPI[key]["max"]


def _fnum(value) -> float | None:
    """Return a finite float or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float('inf'), float('-inf')):
        return None
    return v


def _fmt(value, spec: str = ".3f") -> str:
    """Format a possibly-missing/None/non-numeric value, never raising.

    Replaces the previous ``f"{d.get(key, 0):.3f}"`` pattern, which only
    substitutes the default when the key is absent — an explicit ``None``
    value (a normal way for an engine to say "not computed") still reaches
    the format spec and raises TypeError. This always resolves through
    ``_fnum`` first and falls back to "—" for anything non-finite/missing.
    """
    v = _fnum(value)
    if v is None:
        return "—"
    return f"{v:{spec}}"
def _format_confidence(value) -> str:
    """Format confidence as percentage string, e.g. 0.9 -> '90%'."""
    if value is None:
        return "—"
    try:
        pct = float(value) * 100
        if pct >= 100:
            return "100%"
        if pct <= 0:
            return "0%"
        return f"{pct:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _format_percent(value) -> str:
    """Format a probability (0..1) as a human-readable percentage with up to
    one decimal, trimming a trailing '.0'. e.g. 0.7366 -> '73.7%', 0.9 -> '90%'.
    """
    if value is None:
        return "—"
    try:
        pct = float(value) * 100
        if pct >= 100:
            return "100%"
        if pct <= 0:
            return "0%"
        text = f"{pct:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}%"
    except (TypeError, ValueError):
        return "—"


def _format_scenario_name(name: str) -> str:
    """Convert forecast_day_2 -> Forecast Day 2."""
    if not name:
        return "—"
    if "forecast_day_" in name:
        day_num = name.split("_")[-1]
        return f"Forecast Day {day_num}"
    return " ".join(word.capitalize() for word in name.split("_"))


def _render_recommendation_card(r: dict, rank: int) -> None:
    """Render a single recommendation as a human-readable card.

    Never exposes internal keys (decision_status, recommendation_status,
    aggregate_probability, raw, etc.) except in optional Technical details.
    """
    st.markdown(f"### Recommendation #{rank}")

    action = r.get("action", "—")
    st.write(f"**Action**: {action}")

    kpi = r.get("affected_kpi", "—")
    kpi_label = kpi.replace("_", " ") if kpi else "—"
    st.write(f"**Affected KPI**: {kpi_label}")

    direction = r.get("direction", "—")
    st.write(f"**Direction**: {direction}")

    confidence = _format_confidence(r.get("confidence"))
    st.write(f"**Confidence**: {confidence}")

    risk = r.get("risk", "—")
    st.write(f"**Risk**: {risk}")

    eff = r.get("expected_effect", {})
    if eff:
        oh_gain = eff.get("oh_gain", eff.get("oh_lift", eff.get("oh_improvement", None)))
        nps_gain = eff.get("nps_gain", eff.get("nps_lift", None))
        parts = []
        if oh_gain is not None:
            parts.append(f"OH gain: {oh_gain}")
        if nps_gain is not None:
            parts.append(f"NPS gain: {nps_gain}")
        if parts:
            st.write(f"**Expected effect**: {'; '.join(parts)}")

    evidence = r.get("evidence", [])
    if evidence:
        ev_summary = "; ".join(str(e) for e in evidence[:2])
        st.write(f"**Evidence**: {ev_summary}")


def _render_technical_details_expander(details: dict, title: str = "Technical details") -> None:
    """Add a Streamlit expander with raw JSON for debugging.

    This is the ONLY place raw dict/JSON is shown to the user.
    All other sections display human-readable text.
    """
    with st.expander(title, expanded=False):
        st.json(details)


def _format_current_state(cs: dict) -> str:
    """Format current state as human-readable text.

    Never exposes internal keys like decision_status, recommendation_status,
    aggregate_probability, etc.
    """
    parts = []
    prob = _fnum(cs.get("aggregate_probability"))
    if prob is not None:
        parts.append(f"Overall decision probability: {_format_percent(prob)}")
    conf = _fnum(cs.get("aggregate_confidence"))
    if conf is not None:
        parts.append(f"Confidence: {_format_percent(conf)}")
    observed = cs.get("observed_metrics")
    if observed:
        labels = [_label_metric(m) for m in observed if m]
        if labels:
            parts.append(f"Observed metrics: {', '.join(labels)}")
    for key in ("operations_health", "nps"):
        if key in cs and cs.get(key) is not None:
            parts.append(f"{_label_metric(key)}: {cs[key]}")
    return "\n".join(parts) if parts else "No state information available"


def _format_forecast_outlook(fo: dict) -> str:
    """Format forecast outlook as human-readable text."""
    parts = []
    scenario_count = fo.get("scenario_count", 0)
    parts.append(f"Scenario count: {scenario_count}")
    horizon = fo.get("horizon_days")
    if horizon is not None:
        unit = "day" if int(horizon) == 1 else "days"
        parts.append(f"Forecast horizon: {int(horizon)} {unit}")
    best = fo.get("best_scenario")
    if best:
        parts.append(f"Best scenario: {_format_scenario_name(str(best))}")
    targets = fo.get("targets")
    if targets:
        target_parts = []
        for key in ("target_oh", "target_operations_health", "target_nps"):
            if targets.get(key) is not None:
                target_parts.append(f"{_label_metric(key)}: {targets[key]}")
        if target_parts:
            parts.append("Targets: " + "; ".join(target_parts))
    return "\n".join(parts) if parts else "No forecast information available"


def _label_metric(key: str) -> str:
    """Convert an internal metric/identifier to a human-readable label.

    ``operations_health`` -> ``Operational Health``, ``target_oh`` -> ``OH``,
    ``aggregate_probability`` -> ``Decision Probability``.
    """
    if not key:
        return ""
    label_map = {
        "operations_health": "Operational Health",
        "operational_health": "Operational Health",
        "aggregate_probability": "Decision Probability",
        "aggregate_confidence": "Confidence",
        "target_operations_health": "Operational Health target",
        "target_oh": "OH target",
        "target_nps": "NPS target",
        "observed_metrics": "Observed metrics",
    }
    if key in label_map:
        return label_map[key]
    return " ".join(word.capitalize() for word in str(key).replace("_", " ").split())


def _format_why_selected(ws: dict, fp: dict | None = None, ps: dict | None = None) -> str:
    """Format 'Why Selected' as human-readable text.

    Rules:
    - Never expose internal keys (re_ranking, decision_status, etc.)
    - If insufficient evidence: clear sentence
    - If actionable: describe why the scenario was preferred
    """
    if ws and ws.get("text"):
        return ws["text"]

    reasons = []
    if fp:
        fp_name = _format_scenario_name(fp.get("name", ""))
        if fp_name != "—":
            reasons.append(f"Scenario {fp_name} ranked highest")
    if ps:
        ps_score = ps.get("score", None)
        if ps_score is not None:
            reasons.append(f"Score: {_fmt(ps_score, '.2f')}")
    if not reasons:
        return "Preference determined by deterministic scenario-ranking policy"
    return ". ".join(reasons) + "."


def _format_decision_changers(dc: dict) -> str:
    """Format decision changers as human-readable text.

    Never exposes internal keys like re_ranking, decision_status, etc.
    """
    parts = []
    re_ranking = dc.get("re_ranking")
    if re_ranking:
        parts.append("A materially different forecast scenario would change the "
                     "preferred scenario via the deterministic ranking policy")
    risk_change = dc.get("risk")
    if risk_change:
        parts.append("A change in aggregate probability or confidence would move "
                     "the canonical risk level and therefore the recommended action")
    return "; ".join(parts) if parts else "No decision changers identified"


def _format_uncertainty(unc: dict) -> str:
    """Format uncertainty as human-readable text, never a raw dict."""
    if not unc:
        return "No uncertainty information available"
    parts = []
    down = _fnum(unc.get("downside"))
    if down is not None:
        parts.append(f"Downside (p05): {_format_percent(down)}")
    up = _fnum(unc.get("upside"))
    if up is not None:
        parts.append(f"Upside (p95): {_format_percent(up)}")
    conf = _fnum(unc.get("confidence"))
    if conf is not None:
        parts.append(f"Confidence: {_format_percent(conf)}")
    samples = unc.get("monte_carlo_samples")
    if samples is not None:
        parts.append(f"Monte Carlo samples: {samples:,}")
    interp = unc.get("interpretation")
    if interp:
        parts.append(str(interp))
    return "; ".join(parts) if parts else "No uncertainty information available"




def extract_adie_display(result: dict) -> tuple[dict, dict]:
    """Extract the (probabilistic, details) display surfaces from an ADIE payload.

    Pure helper — no streamlit interaction — so the GUI rendering contract
    can be unit-tested without a browser session.

    Resolution priority (backward compatible):
      1. ``decision_intelligence.details`` / ``decision_intelligence.probabilistic``
      2. legacy ``decision.details`` (composed package under ``decision``)
      3. raw ``decision`` probe (executive fallback only)

    Never fabricates: unavailable surfaces resolve to empty dicts.
    """
    di = result.get("decision_intelligence") or {}
    decision = result.get("decision") or {}

    if isinstance(di, dict) and di.get("details"):
        details = dict(di["details"])
        probabilistic = di.get("probabilistic") or (decision.get("probabilistic") or {})
    elif isinstance(decision, dict) and decision.get("details"):
        details = dict(decision["details"])
        probabilistic = decision.get("probabilistic") or {}
    elif isinstance(di, dict) and di.get("probabilistic"):
        details = dict(di.get("details") or {})
        probabilistic = dict(di["probabilistic"])
    else:
        details = {}
        # Legacy probe: the whole decision dict may itself be the
        # probabilistic surface (executive fields at top level).
        potential = (
            di.get("package")
            if isinstance(di, dict) and isinstance(di.get("package"), dict)
            else decision
        )
        probabilistic = dict(potential or {})

    return probabilistic, details


def render() -> None:
    c.page_title("ADIE Decision", help_text="Canonical Decision Intelligence V3 output")

    from gui import model_selection as ms

    option = ms.render_model_selector(feature="adie")
    family = option.family if option is not None else None

    scenarios = svc.list_scenarios()
    scenario_names = [s["id"] for s in scenarios]

    with st.form("adie_form"):
        st.markdown("#### Decision Inputs")
        c1, c2 = st.columns(2)
        horizon = c1.slider("Horizon (days)", 1, 30, 5)
        scenario = c2.selectbox("Scenario", options=scenario_names,
                                index=scenario_names.index("baseline") if "baseline" in scenario_names else 0)
        q1, q2, q3 = st.columns(3)
        q_lo, q_hi = _bounds("quality")
        cpt_lo, cpt_hi = _bounds("competency")
        a_lo, a_hi = _bounds("attendance")
        r_lo, r_hi = _bounds("release")
        t_lo, t_hi = _bounds("transfer")
        oh_lo, oh_hi = _bounds("operations_health")
        nps_lo, nps_hi = _bounds("nps")
        quality = q1.number_input("Quality %", q_lo, q_hi, ct.kpi_default("quality"))
        competency = q2.number_input("Competency %", cpt_lo, cpt_hi, ct.kpi_default("competency"))
        attendance = q3.number_input("Attendance %", a_lo, a_hi, ct.kpi_default("attendance"))
        release = q1.number_input("Release Rate %", r_lo, r_hi, ct.kpi_default("release"))
        transfer = q2.number_input("Transfer Rate %", t_lo, t_hi, ct.kpi_default("transfer"))
        ops_health = q3.number_input("Operational Health %", oh_lo, oh_hi, ct.kpi_default("operations_health"))
        nps = q1.number_input(ct.NPS_INPUT_LABEL, nps_lo, nps_hi,
                              ct.kpi_default("nps"), help=ct.NPS_INPUT_HELP)

        # Optional targets: enable recommendations, the Monte Carlo success
        # rate and per-metric target probabilities (never fabricated).
        c4, c5 = st.columns(2)
        use_targets = c4.checkbox(
            "Define decision targets", value=True,
            help="Targets activate the Forecast AI recommendation engine, the "
                 "Monte Carlo success rate (P(OH >= target)) and per-metric "
                 "target probabilities.",
        )
        if use_targets:
            t1c, t2c = st.columns(2)
            target_oh = t1c.number_input(
                "Target Operational Health %", ct.OH_MIN, ct.OH_MAX,
                ct.kpi_default("operations_health"),
            )
            target_nps = t2c.number_input(
                "Target NPS", ct.NPS_MIN, ct.NPS_MAX, ct.kpi_default("nps"),
            )
        else:
            target_oh = None
            target_nps = None
        submitted = st.form_submit_button("Run ADIE Decision", type="primary",
                                          disabled=option is None)

    if submitted and family:
        state = {
            "quality": float(quality),
            "competency": float(competency),
            "attendance": float(attendance),
            "release": float(release),
            "transfer": float(transfer),
            "operations_health": float(ops_health),
            "nps": float(nps),
        }
        with st.spinner("Running ADIE V3 decision…"):
            payload = c.guarded(
                svc.adie_decision, state, horizon, scenario, family,
                target_oh=target_oh if use_targets else None,
                target_nps=target_nps if use_targets else None,
            )
        if payload:
            st.session_state["adie_result"] = payload

    result = st.session_state.get("adie_result")
    if not result:
        c.empty_state("Run an ADIE decision to see canonical V3 output.", icon="🧠")
        return

    st.divider()
    ms.render_result_model((result.get("forecast") or {}).get("active_family"), option)
    if not result.get("success"):
        st.error("ADIE decision failed")
        c.error_alert(result.get("errors") or [])
        c.raw_json_expander(result)
        return

    # ---- Extract detail ----
    probabilistic, details = extract_adie_display(result)

    # Executive Decision section (from probabilistic package)
    _render_executive_decision(probabilistic, details.get("recommendations", []))

    # Detail sections from enriched detail
    if details:
        # Forecast Overview + Table
        _render_forecast_overview(details.get("forecast_summary", {}))
        _render_forecast_table(details.get("forecast_summary", {}).get("per_day_table", []))

        # Bayesian Detail
        _render_bayesian_detail(details.get("bayesian_detail", {}))

        # Monte Carlo Detail
        _render_mc_detail(details.get("monte_carlo_detail", {}))

        # Risk Detail
        _render_risk_detail(details.get("risk_detail", {}))

        # Scenario Comparison
        _render_scenario_comparison(
            details.get("scenario_comparison", []),
            decision_status=details.get("decision_status"),
        )

        # Sensitivity Detail
        _render_sensitivity_detail(details.get("sensitivity_detail", {}))

        # Trend Detail
        _render_trend_detail(details.get("trend_detail", {}))

        # Agreement
        _render_agreement_detail(details.get("agreement", {}))

        # Top 3 Recommendations
        _render_recommendations(
            details.get("recommendations", []),
            recommendation_status=details.get("recommendation_status"),
        )

        # Explanation
        _render_explanation(details.get("explanation", {}))
    else:
        st.info("Enriched detail not available (run via Forecast path for full detail).")

    # Forecast context
    st.markdown("---")
    st.markdown("#### Forecast Context")
    fc = result.get("forecast") or {}
    if fc:
        st.caption(f"H{fc.get('horizon')} · scenario `{fc.get('scenario')}`")
        timeline = fc.get("timeline") or []
        if timeline:
            last = timeline[-1]
            st.write(f"End OH: {last.get('operations_health')} · End NPS: {last.get('nps')}")
    else:
        st.caption("No forecast context attached.")

    c.warning_alert(result.get("warnings") or [])
    c.raw_json_expander(result.get("raw") or result)

    # ---- Analytics ----
    st.divider()
    from gui.analytics import adie as _aa
    _aa.render_analytics(st, result)


def _render_executive_decision(package: dict, recommendations: list[dict] | None = None) -> None:
    """Render the core executive decision metrics.

    Recommendation consistency (Phase 16 fix): the Executive Decision shows
    the top-ranked recommendation. If no genuine recommendations exist, it
    shows "No recommendation available" rather than claiming a recommendation
    that was not actually produced.
    """
    st.markdown("### Executive Decision")

    decision_status = package.get("decision_status")
    insufficient = package.get("abstain") is True or str(package.get("risk", "")).upper() == "ABSTAIN" or decision_status == "insufficient_evidence"

    recs = recommendations or []
    if insufficient:
        rec_label = "None — insufficient evidence"
        rec_help = "Canonical decision withheld: recommendation evidence is insufficient"
    elif recs:
        top = recs[0]
        rec_label = f"#{top.get('rank', 1)} · {top.get('action', '—')}"
        rec_help = f"Matches Top Recommendation (KPI: {top.get('affected_kpi') or '—'})"
    else:
        rec_label = "No recommendation available"
        rec_help = "No genuine recommendations produced by the Forecast AI engines"

    cols = st.columns(4)
    with cols[0]:
        st.metric("Decision Status", "Insufficient evidence" if insufficient else "Available",
                  help="Canonical decision status. Insufficient evidence withholds the decision.")
    with cols[1]:
        risk = package.get("risk", "—")
        color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "ABSTAIN": "gray"}.get(str(risk), "gray")
        st.metric("Risk Decision", str(risk) if risk else "—")
        st.markdown(f"<span style='color:{color}'>●</span>", unsafe_allow_html=True)
    with cols[2]:
        prob = package.get("probability")
        st.metric("Probability", f"{prob:.3f}" if isinstance(prob, (int, float)) else "—")
    with cols[3]:
        conf = package.get("confidence")
        st.metric("Confidence", _format_confidence(conf))

    cols2 = st.columns(4)
    with cols2[0]:
        exp = package.get("expected")
        st.metric("Expected", f"{exp:.3f}" if isinstance(exp, (int, float)) else "—")
    with cols2[1]:
        down = package.get("downside")
        st.metric("Downside (p05)", f"{down:.3f}" if isinstance(down, (int, float)) else "—")
    with cols2[2]:
        up = package.get("upside")
        st.metric("Upside (p95)", f"{up:.3f}" if isinstance(up, (int, float)) else "—")
    with cols2[3]:
        risk_score = package.get("risk_score")
        st.metric("Risk Score", f"{risk_score:.3f}" if isinstance(risk_score, (int, float)) else "—")

    if insufficient:
        st.warning("⚠️ Decision evidence is insufficient — the engine abstains. "
                   "Risk decision: ABSTAIN. Recommendations: None — insufficient evidence.")
    elif package.get("abstain"):
        st.warning("⚠️ ADIE abstained on this decision.")


def _render_forecast_overview(fc_summary: dict) -> None:
    """Render forecast summary ranges.

    Phase 16 fix: the OH/NPS min/max/expected shown here are the POINT forecast
    range (min/max across the forecast days) — explicitly labelled as such, not
    presented as a confidence interval. The probabilistic interval (p05/p95 of
    the single Monte Carlo) is shown separately when present.
    """
    st.markdown("### Forecast Overview")
    if not fc_summary:
        st.caption("No forecast summary available.")
        return

    st.caption("OH / NPS values below are the **point forecast range** "
               "(min/max across forecast days), not a confidence interval.")

    char = fc_summary.get("character") or {}
    if char.get("flat_nps") or char.get("flat_oh"):
        note = char.get("note") or (
            "Per-day OH/NPS are point forecasts and are identical across days; "
            "uncertainty is expressed as horizon confidence decay."
        )
        st.info(f"ℹ️ {note}")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Operations Health — Point Forecast Range**")
        oh = fc_summary.get("oh_range", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Min", f"{oh.get('min'):.1f}" if oh.get("min") is not None else "—")
        c2.metric("Expected", f"{oh.get('expected'):.1f}" if oh.get("expected") is not None else "—")
        c3.metric("Max", f"{oh.get('max'):.1f}" if oh.get("max") is not None else "—")

    with cols[1]:
        st.markdown("**NPS — Point Forecast Range**")
        nps = fc_summary.get("nps_range", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Min", f"{nps.get('min'):.1f}" if nps.get("min") is not None else "—")
        c2.metric("Expected", f"{nps.get('expected'):.1f}" if nps.get("expected") is not None else "—")
        c3.metric("Max", f"{nps.get('max'):.1f}" if nps.get("max") is not None else "—")

    # Probabilistic interval (p05/p95) — shown separately, correctly labelled.
    pi = fc_summary.get("probabilistic_interval", {}) or {}
    if pi.get("oh_p05") is not None or pi.get("oh_p95") is not None:
        lo = f"{pi.get('oh_p05'):.2f}" if pi.get("oh_p05") is not None else "—"
        hi = f"{pi.get('oh_p95'):.2f}" if pi.get("oh_p95") is not None else "—"
        st.metric("OH Probabilistic Interval (p05–p95)", f"{lo} – {hi}",
                  help=pi.get("label", "p05/p95 of the single Monte Carlo"))

    # Best/Worst/Expected day (actual ADIE ranking, not forced to day 1)
    cols3 = st.columns(3)
    best = fc_summary.get("best_day", {})
    worst = fc_summary.get("worst_day", {})
    expected = fc_summary.get("expected_day", {})
    with cols3[0]:
        st.metric("Best Day OH", f"{best.get('oh'):.1f}" if best.get("oh") is not None else "—")
        st.caption(f"Rank {best.get('rank', best.get('day_index', '?'))} · {best.get('name') or '—'} · NPS: {best.get('nps', '—')}")
    with cols3[1]:
        st.metric("Worst Day OH", f"{worst.get('oh'):.1f}" if worst.get("oh") is not None else "—")
        st.caption(f"Rank {worst.get('rank', worst.get('day_index', '?'))} · {worst.get('name') or '—'} · NPS: {worst.get('nps', '—')}")
    with cols3[2]:
        st.metric("Expected Day OH", f"{expected.get('oh'):.1f}" if expected.get("oh") is not None else "—")
        st.caption(f"Rank {expected.get('rank', expected.get('day_index', '?'))} · {expected.get('name') or '—'} · NPS: {expected.get('nps', '—')}")

    st.caption(f"Scenario count: {fc_summary.get('scenario_count', 0)} · Horizon: {fc_summary.get('horizon_days', '?')} days")


def _render_forecast_table(per_day: list[dict]) -> None:
    """Render per-day forecast table.

    ``_predicted`` is displayed as the real boolean from the source
    (True/False), never NULL when present. Semantically unambiguous columns:
      - ``nps``            : NPS on the -100..100 scale
      - ``expected_score`` : mean 0..10 survey score
      - ``score_p05/p95``  : 0..10 score quantiles
      - ``nps_p05/p95``    : -100..100 NPS quantiles (Monte Carlo)
    """
    st.markdown("### Forecast Table (Per Day)")
    if not per_day:
        st.caption("No per-day forecast data.")
        return

    df = pd.DataFrame(per_day)
    if df.empty:
        st.caption("Empty forecast table.")
        return

    # Display-only copy with explicit, unambiguous column labels.
    view = df.copy()
    rename = {
        "expected_score": "Expected Score (0-10)",
        "score_p05": "Score P05 (0-10)",
        "score_p95": "Score P95 (0-10)",
        "nps_p05": "NPS P05 (-100..100)",
        "nps_p95": "NPS P95 (-100..100)",
        "nps": "NPS (-100..100)",
        "oh": "OH",
        "confidence": "Confidence",
        "risk": "Risk",
        "day": "Day",
    }
    view = view.rename(columns=rename)
    cols_order = [
        "Day", "OH", "NPS (-100..100)", "Expected Score (0-10)",
        "Score P05 (0-10)", "Score P95 (0-10)",
        "NPS P05 (-100..100)", "NPS P95 (-100..100)",
        "Confidence", "Risk", "_predicted",
    ]
    display_cols = [c for c in cols_order if c in view.columns]
    view = view[display_cols] if display_cols else view
    # Show _predicted as explicit True/False (never None when source has it).
    if "_predicted" in view.columns:
        view["_predicted"] = view["_predicted"].map(
            lambda v: "True" if v is True else ("False" if v is False else "—")
        )
    st.dataframe(view, width="stretch")
    st.caption("`NPS` is on the -100..100 scale; `Expected Score` and `Score P05/P95` are 0..10 survey scores; `NPS P05/P95` are Monte Carlo NPS quantiles (-100..100). `_predicted=True` → forecast-generated day; `_predicted=False` → observed day.")


def _render_bayesian_detail(bayesian: dict) -> None:
    """Render Bayesian posterior detail."""
    st.markdown("### Bayesian Analysis")
    if not bayesian:
        st.caption("No Bayesian detail available.")
        return

    cols = st.columns(2)
    with cols[0]:
        st.metric("Decision Probability", _format_percent(bayesian.get('decision_probability')))
        st.metric("Posterior Mean", _format_percent(bayesian.get('posterior_ranges', {}).get('mean')))
    with cols[1]:
        st.metric("Confidence", _format_percent(bayesian.get('confidence')))
        st.metric("Posterior Std", _fmt(bayesian.get('posterior_ranges', {}).get('std')))

    cred = bayesian.get("posterior_ranges", {}).get("credible_interval", {})
    lo = _format_percent(cred.get('lower'))
    hi = _format_percent(cred.get('upper'))
    level = cred.get('level')
    level_txt = _format_percent(level) if isinstance(level, (int, float)) else "—"
    st.metric("Credible Interval", f"{lo} – {hi} ({level_txt})")

    # Target probabilities
    pot = bayesian.get("probability_of_target")
    if pot:
        st.markdown("**Per-Metric Target Probabilities**")
        for k, v in pot.items():
            if isinstance(v, dict):
                target = v.get("target")
                prob = v.get("probability")
                if prob is None:
                    st.write(f"- **{_label_metric(str(k))}**: unavailable")
                else:
                    st.write(f"- **{_label_metric(str(k))}**: {_format_percent(prob)}")
            else:
                st.write(f"- **{_label_metric(str(k))}**: {_format_percent(v)}")

    # NPS 0-10 distribution (chart + table)
    nps_dist = bayesian.get("nps_0_10_distribution")
    if nps_dist:
        st.markdown("**NPS 0–10 Posterior Distribution**")
        # Normalize to 11 rows: Score 0..10 -> Probability.
        scores = []
        if isinstance(nps_dist, dict):
            for key, prob in nps_dist.items():
                score = None
                if isinstance(key, int):
                    score = key
                elif isinstance(key, str) and key.isdigit():
                    score = int(key)
                elif isinstance(key, str) and key.lower().startswith("score_"):
                    try:
                        score = int(key.split("_", 1)[1])
                    except (ValueError, IndexError):
                        score = None
                if score is not None and 0 <= score <= 10 and _fnum(prob) is not None:
                    scores.append((score, float(prob)))
        if scores:
            rows = []
            prob_by_score = {s: p for s, p in scores}
            for score in range(11):
                rows.append({"Score": score, "Probability": prob_by_score.get(score, 0.0)})
            dist_df = pd.DataFrame(rows)
            st.dataframe(dist_df, width="stretch")
            st.bar_chart(dist_df.set_index("Score")[["Probability"]])
        else:
            st.caption("NPS 0–10 distribution present but not in a renderable 0–10 form.")
    else:
        st.caption("No NPS 0–10 posterior distribution available.")

    st.caption(bayesian.get("interpretation", ""))


def _render_mc_detail(mc: dict) -> None:
    """Render Monte Carlo detail.

    The Monte Carlo package contains TWO distinct statistics that must not be
    conflated:

      1. Binary target-attainment outcome. ``success_count`` / ``failure_count``
         / ``success_percentage`` / ``failure_percentage`` describe whether the
         simulated OH outcome satisfies the target (P(OH >= target_oh)). This is
         a genuine binary success/failure rate.

      2. Continuous Monte Carlo score. ``expected_value`` / ``p05`` / ``p50`` /
         ``p95`` / ``uncertainty`` are the percentiles and spread of the
         underlying continuous OH distribution (a normal draw, not a 0/1
         outcome). They are NOT the binary expectation and must never be
         presented as the expected binary success value.

    The two sections are rendered separately so the binary success rate is never
    confused with the continuous score statistics.
    """
    st.markdown("### Monte Carlo Analysis")
    if not mc:
        st.caption("No Monte Carlo detail available.")
        return

    success_count = mc.get("success_count")
    failure_count = mc.get("failure_count")
    success_pct = mc.get("success_percentage")
    failure_pct = mc.get("failure_percentage")

    # ---- Binary outcome (target attainment) ----
    st.markdown("#### Binary Outcome")
    has_binary = success_count is not None or failure_count is not None \
        or success_pct is not None or failure_pct is not None
    if not has_binary:
        st.caption("Binary success/failure not computed: no target defined.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples", f"{mc.get('total_samples', 0):,.0f}"
                  if mc.get("total_samples") is not None else "—")
        c2.metric("Success", f"{success_count:,}" if success_count is not None else "Unavailable")
        c3.metric("Failure", f"{failure_count:,}" if failure_count is not None else "Unavailable")
        c4.metric("Success Rate", f"{success_pct:.1f}%" if success_pct is not None else "Unavailable")

        c5, c6 = st.columns(2)
        c5.metric("Failure Rate", f"{failure_pct:.1f}%" if failure_pct is not None else "Unavailable")
        c6.metric("Total Samples", f"{mc.get('total_samples', 0):,.0f}"
                  if mc.get("total_samples") is not None else "—")
    if mc.get("success_definition"):
        st.caption(f"Success definition: {mc['success_definition']}")

    # ---- Continuous score ----
    st.markdown("#### Continuous Monte Carlo Score")
    st.caption("Percentiles and spread of the underlying continuous OH "
               "distribution (NOT a binary success value).")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Expected Score", _fmt(mc.get('expected_value')))
    with cols[1]:
        st.metric("Uncertainty", _fmt(mc.get('uncertainty')))
    with cols[2]:
        st.metric("p05", _fmt(mc.get('p05')))
    with cols[3]:
        st.metric("p50", _fmt(mc.get('p50')))

    cols2 = st.columns(4)
    with cols2[0]:
        st.metric("p95", _fmt(mc.get('p95')))
    with cols2[3]:
        dist = mc.get("distribution_bins", [])
        if dist:
            df = pd.DataFrame(dist)
            st.bar_chart(df.set_index("bin_start")[["probability"]])

    if mc.get("interpretation"):
        st.caption(mc.get("interpretation", ""))


def _render_risk_detail(risk: dict) -> None:
    """Render risk detail."""
    st.markdown("### Risk Analysis")
    if not risk:
        st.caption("No risk detail available.")
        return

    insufficient = risk.get("status") == "insufficient_evidence" or str(risk.get("level", "")).upper() == "ABSTAIN"

    if insufficient:
        st.warning("⚠️ Risk decision: **ABSTAIN** — canonical decision is withheld "
                   "because decision evidence is insufficient.")
        reason = risk.get("reason")
        if reason:
            st.caption(reason)
        raw = risk.get("raw") or {}
        cols = st.columns(3)
        with cols[0]:
            st.metric("Risk Level", "ABSTAIN")
        with cols[1]:
            st.caption("Raw diagnostic score (not a decision):")
            st.metric("Raw Risk Score", _fmt(raw.get("score")))
        with cols[2]:
            st.caption("Raw inputs preserved as diagnostic metadata only.")
            st.metric("Raw Confidence", _fmt(raw.get("confidence")))
        cols2 = st.columns(3)
        with cols2[0]:
            st.metric("Raw Downside", _fmt(raw.get("downside")))
        with cols2[1]:
            st.metric("Raw Upside", _fmt(raw.get("upside")))
        with cols2[2]:
            st.metric("Abstain", "Yes")
        return

    cols = st.columns(3)
    with cols[0]:
        level = risk.get("level", "—")
        color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "ABSTAIN": "gray"}.get(str(level), "gray")
        st.metric("Risk Level", str(level))
        st.markdown(f"<span style='color:{color};font-size:2rem'>●</span>", unsafe_allow_html=True)
    with cols[1]:
        st.metric("Risk Score", _fmt(risk.get('score')))
    with cols[2]:
        st.metric("Confidence", _fmt(risk.get('confidence')))

    cols2 = st.columns(3)
    with cols2[0]:
        st.metric("Downside", _fmt(risk.get('downside')))
    with cols2[1]:
        st.metric("Upside", _fmt(risk.get('upside')))
    with cols2[2]:
        st.metric("Abstain", "Yes" if risk.get("abstain") else "No")


def _render_scenario_comparison(scenarios: list[dict], *, decision_status: str | None = None) -> None:
    """Render scenario comparison table (actual ADIE rank + score + factors)."""
    st.markdown("### Scenario Comparison")
    if not scenarios:
        st.caption("No scenario comparison available.")
        return

    if decision_status == "insufficient_evidence":
        st.info("ℹ️ **Forecast ranking only — insufficient decision evidence.** "
                "The ranked order below is a forecast-ranking result, not an "
                "actionable decision/recommendation.")

    df = pd.DataFrame(scenarios)
    if df.empty:
        st.caption("Empty scenario list.")
        return

    # Show key columns: rank/score first, then values, then ranking factors.
    key_cols = ["rank", "score", "name", "oh", "nps", "probability", "confidence",
                "expected", "p05", "p50", "p95", "risk_severity", "_predicted",
                "ranking_factors"]
    display = [c for c in key_cols if c in df.columns]
    view = df[display] if display else df
    # Drop columns that carry no data for any scenario (e.g. probability/p05/
    # p50/p95 when forecast-day scenarios have no per-day stats) — a blank
    # column is a display defect, not information.
    view = view.dropna(axis=1, how="all")
    st.dataframe(view, width="stretch")

    st.caption("Rank = deterministic ADIE scenario policy order (score = "
               "weighted normalized forecast evidence). It is the real ranking, "
               "not a forced forecast-day-1 order.")

    # Show ranking evidence for the top scenario.
    if scenarios:
        top = scenarios[0]
        if top.get("ranking_factors"):
            st.write(f"**Top ({top.get('name')}, rank {top.get('rank', 1)}, "
                     f"score {top.get('score', '—')}) ranking factors:** "
                     f"{', '.join(str(f) for f in top['ranking_factors'])}")


def _render_sensitivity_detail(sens: dict) -> None:
    """Render sensitivity detail."""
    st.markdown("### Sensitivity Analysis")
    if not sens:
        st.caption("No sensitivity detail available.")
        return

    metrics = sens.get("metrics", [])
    if not metrics:
        st.caption("No metric sensitivities.")
        return

    df = pd.DataFrame(metrics)
    st.dataframe(df, width="stretch")

    ranking = sens.get("ranking", [])
    if ranking:
        st.caption(f"Ranking: {' > '.join(ranking)}")


def _render_trend_detail(trend: dict) -> None:
    """Render trend detail."""
    st.markdown("### Trend Analysis")
    if not trend:
        st.caption("No trend detail available.")
        return

    direction = trend.get("direction", "stable")
    st.metric("Overall Direction", direction.title())

    sp = trend.get("strongest_positive")
    sn = trend.get("strongest_negative")
    if sp:
        st.metric("Strongest Positive", f"{sp.get('metric')}: {_fmt(sp.get('change'), '.2f')}")
    if sn:
        st.metric("Strongest Negative", f"{sn.get('metric')}: {_fmt(sn.get('change'), '.2f')}")

    analyses = trend.get("analyses", [])
    if analyses:
        df = pd.DataFrame(analyses)
        st.dataframe(df, width="stretch")


def _render_agreement_detail(agreement: dict) -> None:
    """Render agreement/conflicts.

    When evidence is insufficient (no computed agreement score/consistency), an
    explicit "insufficient evidence" state is shown rather than an apparently
    confident block of —/0 metrics.
    """
    st.markdown("### Agreement & Conflicts")
    if not agreement:
        st.caption("No agreement detail available.")
        return

    if agreement.get("available") is False or agreement.get("status") == "insufficient_evidence":
        st.warning("Insufficient evidence to compute agreement/consistency.")
        # Structured, explicit insufficient-evidence state — never a fake
        # conflict_count=0 (conflict analysis did not run).
        st.json({
            "status": agreement.get("status", "insufficient_evidence"),
            "agreement_score": agreement.get("score"),
            "category_consistency": agreement.get("category_consistency"),
            "conflict_count": agreement.get("conflict_count"),
            "reason": agreement.get("reason") or "No recommendation evidence available",
        })
        reason = agreement.get("reason")
        if reason:
            st.caption(reason)
        return

    cols = st.columns(3)
    with cols[0]:
        st.metric("Agreement Score", _fmt(agreement.get('score'), '.2f'))
    with cols[1]:
        st.metric("Category Consistency", _fmt(agreement.get('category_consistency'), '.2f'))
    with cols[2]:
        st.metric("Conflict Count", agreement.get("conflict_count", 0))

    conflicts = agreement.get("conflicts", [])
    if conflicts:
        st.markdown("**Conflicts**")
        for c in conflicts[:5]:
            st.warning(str(c))


def _render_recommendations(recs: list[dict], *, recommendation_status: str | None = None) -> None:
    """Render top 3 recommendations with human-readable format.

    Rules:
    1. If recommendation_status == insufficient_evidence: show clear sentence, not dict
    2. If no recs: show "No recommendations available"
    3. If recs exist: render each as human-readable card
    4. Internal keys never exposed at top level
    """
    st.markdown("### Top 3 Recommendations")
    # Handle insufficient evidence case
    if recommendation_status == "insufficient_evidence":
        st.caption("Decision withheld — insufficient evidence. No recommendations "
                   "are produced because decision/recommendation evidence is unavailable.")
        return
    if not recs:
        st.caption("No recommendations available.")
        return
    for idx, r in enumerate(recs[:3], start=1):
        _render_recommendation_card(r, idx)


def _render_explanation(explanation: dict) -> None:
    """Render the decision explanation as human-readable prose.

    The user-facing surface exposes only:
      - Recommendation
      - Why
      - Preferred Forecast
      - Risk
      - Confidence
      - What Would Change the Decision

    Internal keys (re_ranking, decision_status, recommendation_status,
    aggregate_probability, raw, ...) are never shown here; the raw structured
    package is available only inside an optional "Technical details" expander.

    Rules:
    1. If decision_status == insufficient_evidence: show the withheld sentence
       and present forecast preference as a separate, non-actionable line.
    2. If actionable: render the recommendation as normal prose.
    3. Confidence formatted as a percentage (0.9 -> 90%).
    4. Scenario names converted to human-readable (forecast_day_2 -> Forecast Day 2).
    """
    if not explanation:
        st.caption("No explanation available.")
        return

    decision_status = explanation.get("decision_status")
    recommended_action = explanation.get("recommended_action") or {}
    insufficient = (
        decision_status == "insufficient_evidence"
        or recommended_action.get("action") == "withheld"
        or recommended_action.get("status") == "insufficient_evidence"
    )

    st.markdown("### Recommendation")
    if insufficient:
        st.caption("Decision withheld — insufficient evidence.")
        # Forecast preference is shown separately, explicitly non-actionable.
        fp = explanation.get("forecast_preference") or {}
        ps = explanation.get("preferred_scenario") or {}
        name = fp.get("name") or ps.get("name")
        st.write(
            f"Forecast preference: **{_format_scenario_name(str(name))}** "
            "(non-actionable)."
        )
        ws = explanation.get("why_selected") or {}
        if ws.get("text"):
            st.caption(ws["text"])
    else:
        action = recommended_action.get("action") or recommended_action.get("recommendation")
        if action:
            st.write(_humanize_action(action))
        else:
            st.caption("No actionable recommendation available.")

    st.markdown("#### Why")
    why_text = _format_why_selected(
        explanation.get("why_selected") or {},
        explanation.get("forecast_preference") or {},
        explanation.get("preferred_scenario") or {},
    )
    st.write(why_text)

    st.markdown("#### Preferred Forecast")
    ps = explanation.get("preferred_scenario") or {}
    name = ps.get("name")
    if name:
        st.write(_format_scenario_name(str(name)))
    else:
        st.write("—")

    st.markdown("#### Risk")
    mr = explanation.get("main_risk") or {}
    if insufficient:
        st.write("**ABSTAIN**")
    else:
        st.write(str(mr.get("level", "—")))

    st.markdown("#### Confidence")
    conf = explanation.get("decision_confidence")
    if conf is None:
        conf = (recommended_action or {}).get("confidence")
    st.write(_format_confidence(conf))

    st.markdown("#### What Would Change the Decision")
    dc = explanation.get("decision_changers") or {}
    st.write(_format_decision_changers(dc))

    # ---- Current State (human-readable, never raw dict) ----
    st.markdown("#### Current State")
    cs = explanation.get("current_state") or {}
    st.write(_format_current_state(cs))

    # ---- Forecast Outlook (human-readable, never raw dict) ----
    st.markdown("#### Forecast Outlook")
    fo = explanation.get("forecast_summary") or {}
    st.write(_format_forecast_outlook(fo))

    # ---- Uncertainty (human-readable) ----
    st.markdown("#### Uncertainty")
    unc = explanation.get("uncertainty") or {}
    st.write(_format_uncertainty(unc))

    # ---- Supporting Evidence ----
    st.markdown("#### Supporting Evidence")
    if insufficient:
        st.write("Canonical decision: ABSTAIN — decision withheld due to insufficient evidence.")
    else:
        supporting = explanation.get("supporting_evidence") or []
        if supporting:
            for line in supporting:
                st.write(f"- {line}")
        else:
            st.write("No supporting evidence available.")

    # Optional raw structured package for debugging only.
    _render_technical_details_expander(dict(explanation))


def _humanize_action(action) -> str:
    """Convert an internal snake_case action to a plain-language sentence."""
    if not action:
        return "No actionable recommendation available."
    text = str(action).replace("_", " ").strip().capitalize()
    return text or "No actionable recommendation available."
