"""ADIE Decision view: render the canonical V3 decision package with rich detail."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from gui import components as c
from gui import services as svc


DEFAULTS = {
    "quality": 87.0,
    "competency": 93.0,
    "attendance": 90.0,
    "release": 60.0,
    "transfer": 9.0,
    "operations_health": 95.0,
    "nps": 82.0,
}


def render() -> None:
    c.page_title("ADIE Decision", help_text="Canonical Decision Intelligence V3 output")

    models = svc.list_models()
    active = svc.STATE.get_active_family()
    if not models:
        c.empty_state("No model families available. Train one first.", icon="🧠")
        return

    options = [m["family"] for m in models if "error" not in m]
    idx = options.index(active) if active in options else 0
    family = st.selectbox("Model family (explicit)", options=options, index=idx)
    if family != active:
        try:
            svc.select_model_family(family)
        except Exception as exc:
            st.error(f"Could not activate {family}: {exc}")
            return

    scenarios = svc.list_scenarios()
    scenario_names = [s["id"] for s in scenarios]

    with st.form("adie_form"):
        st.markdown("#### Decision Inputs")
        c1, c2 = st.columns(2)
        horizon = c1.slider("Horizon (days)", 1, 30, 5)
        scenario = c2.selectbox("Scenario", options=scenario_names,
                                index=scenario_names.index("baseline") if "baseline" in scenario_names else 0)
        q1, q2, q3 = st.columns(3)
        quality = q1.number_input("Quality %", 0.0, 100.0, DEFAULTS["quality"])
        competency = q2.number_input("Competency %", 0.0, 100.0, DEFAULTS["competency"])
        attendance = q3.number_input("Attendance %", 0.0, 100.0, DEFAULTS["attendance"])
        release = q1.number_input("Release Rate %", 0.0, 100.0, DEFAULTS["release"])
        transfer = q2.number_input("Transfer Rate %", 0.0, 100.0, DEFAULTS["transfer"])
        ops_health = q3.number_input("Operational Health %", 0.0, 120.0, DEFAULTS["operations_health"])
        nps = q1.number_input("NPS", -100.0, 100.0, DEFAULTS["nps"])
        submitted = st.form_submit_button("Run ADIE Decision", type="primary")

    if submitted:
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
            payload = c.guarded(svc.adie_decision, state, horizon, scenario, family)
        if payload:
            st.session_state["adie_result"] = payload

    result = st.session_state.get("adie_result")
    if not result:
        c.empty_state("Run an ADIE decision to see canonical V3 output.", icon="🧠")
        return

    st.divider()
    if not result.get("success"):
        st.error("ADIE decision failed")
        c.error_alert(result.get("errors") or [])
        c.raw_json_expander(result)
        return

    # ---- Extract detail ----
    # Priority: decision_intelligence -> decision.details -> decision (legacy)
    di = result.get("decision_intelligence") or {}
    details = di.get("details") or result.get("decision", {}).get("details") or {}
    package = di.get("package") or result.get("decision", {}).get("probabilistic", {}) or result.get("decision", {})

    # Executive Decision section (from probabilistic package)
    _render_executive_decision(package)

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
        _render_scenario_comparison(details.get("scenario_comparison", []))

        # Sensitivity Detail
        _render_sensitivity_detail(details.get("sensitivity_detail", {}))

        # Trend Detail
        _render_trend_detail(details.get("trend_detail", {}))

        # Agreement
        _render_agreement_detail(details.get("agreement", {}))

        # Top 3 Recommendations
        _render_recommendations(details.get("recommendations", []))

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


def _render_executive_decision(package: dict) -> None:
    """Render the core executive decision metrics."""
    st.markdown("### Executive Decision")
    
    cols = st.columns(4)
    with cols[0]:
        rec = package.get("recommendation", "—")
        st.metric("Recommendation", str(rec))
    with cols[1]:
        risk = package.get("risk", "—")
        color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(risk, "gray")
        st.metric("Risk Level", str(risk))
        st.markdown(f"<span style='color:{color}'>●</span>", unsafe_allow_html=True)
    with cols[2]:
        prob = package.get("probability")
        st.metric("Probability", f"{prob:.3f}" if isinstance(prob, (int, float)) else "—")
    with cols[3]:
        conf = package.get("confidence")
        st.metric("Confidence", f"{conf:.3f}" if isinstance(conf, (int, float)) else "—")

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

    if package.get("abstain"):
        st.warning("⚠️ ADIE abstained on this decision.")


def _render_forecast_overview(fc_summary: dict) -> None:
    """Render forecast summary ranges."""
    st.markdown("### Forecast Overview")
    if not fc_summary:
        st.caption("No forecast summary available.")
        return

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Operations Health Range**")
        oh = fc_summary.get("oh_range", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Min", f"{oh.get('min'):.1f}" if oh.get("min") is not None else "—")
        c2.metric("Expected", f"{oh.get('expected'):.1f}" if oh.get("expected") is not None else "—")
        c3.metric("Max", f"{oh.get('max'):.1f}" if oh.get("max") is not None else "—")

    with cols[1]:
        st.markdown("**NPS Range**")
        nps = fc_summary.get("nps_range", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Min", f"{nps.get('min'):.1f}" if nps.get("min") is not None else "—")
        c2.metric("Expected", f"{nps.get('expected'):.1f}" if nps.get("expected") is not None else "—")
        c3.metric("Max", f"{nps.get('max'):.1f}" if nps.get("max") is not None else "—")

    # Best/Worst day
    cols3 = st.columns(3)
    best = fc_summary.get("best_day", {})
    worst = fc_summary.get("worst_day", {})
    expected = fc_summary.get("expected_day", {})
    with cols3[0]:
        st.metric("Best Day OH", f"{best.get('oh'):.1f}" if best.get("oh") is not None else "—")
        st.caption(f"Day {best.get('day_index', 1)} · NPS: {best.get('nps', '—')}")
    with cols3[1]:
        st.metric("Worst Day OH", f"{worst.get('oh'):.1f}" if worst.get("oh") is not None else "—")
        st.caption(f"Day {worst.get('day_index', '?')} · NPS: {worst.get('nps', '—')}")
    with cols3[2]:
        st.metric("Expected Day OH", f"{expected.get('oh'):.1f}" if expected.get("oh") is not None else "—")
        st.caption(f"Day {expected.get('day_index', 1)} · NPS: {expected.get('nps', '—')}")

    st.caption(f"Scenario count: {fc_summary.get('scenario_count', 0)} · Horizon: {fc_summary.get('horizon_days', '?')} days")


def _render_forecast_table(per_day: list[dict]) -> None:
    """Render per-day forecast table."""
    st.markdown("### Forecast Table (Per Day)")
    if not per_day:
        st.caption("No per-day forecast data.")
        return

    df = pd.DataFrame(per_day)
    if df.empty:
        st.caption("Empty forecast table.")
        return

    # Ensure key columns
    cols_order = ["day", "oh", "nps", "confidence", "risk", "_predicted"]
    display_cols = [c for c in cols_order if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True)


def _render_bayesian_detail(bayesian: dict) -> None:
    """Render Bayesian posterior detail."""
    st.markdown("### Bayesian Analysis")
    if not bayesian:
        st.caption("No Bayesian detail available.")
        return

    cols = st.columns(2)
    with cols[0]:
        st.metric("Decision Probability", f"{bayesian.get('decision_probability', 0):.3f}")
        st.metric("Posterior Mean", f"{bayesian.get('posterior_ranges', {}).get('mean', 0):.3f}")
    with cols[1]:
        st.metric("Confidence", f"{bayesian.get('confidence', 0):.3f}")
        st.metric("Posterior Std", f"{bayesian.get('posterior_ranges', {}).get('std', 0):.3f}")

    cred = bayesian.get("posterior_ranges", {}).get("credible_interval", {})
    st.metric("Credible Interval", f"[{cred.get('lower', 0):.3f}, {cred.get('upper', 0):.3f}] ({cred.get('level', 0)*100:.0f}%)")

    # Target probabilities
    pot = bayesian.get("probability_of_target")
    if pot:
        st.markdown("**Per-Metric Target Probabilities**")
        for k, v in pot.items():
            if isinstance(v, dict):
                st.write(f"- **{k}**: target={v.get('target')}, prob={v.get('probability')}")

    # NPS 0-10 distribution
    nps_dist = bayesian.get("nps_0_10_distribution")
    if nps_dist:
        st.markdown("**NPS 0–10 Posterior Distribution**")
        dist_df = pd.DataFrame([{"Score": i, "Probability": p} for i, p in nps_dist.items()])
        st.bar_chart(dist_df.set_index("Score"))

    st.caption(bayesian.get("interpretation", ""))


def _render_mc_detail(mc: dict) -> None:
    """Render Monte Carlo detail."""
    st.markdown("### Monte Carlo Analysis")
    if not mc:
        st.caption("No Monte Carlo detail available.")
        return

    cols = st.columns(4)
    with cols[0]:
        st.metric("Samples", mc.get("total_samples", "—"))
    with cols[1]:
        st.metric("Success Count", mc.get("success_count", "—"))
        st.metric("Failure Count", mc.get("failure_count", "—"))
    with cols[2]:
        st.metric("Success %", f"{mc.get('success_percentage', 0):.1f}%")
        st.metric("Failure %", f"{mc.get('failure_percentage', 0):.1f}%")
    with cols[3]:
        st.metric("Expected Value", f"{mc.get('expected_value', 0):.3f}")
        st.metric("Uncertainty", f"{mc.get('uncertainty', 0):.3f}")

    cols2 = st.columns(4)
    with cols2[0]:
        st.metric("p05", f"{mc.get('p05', 0):.3f}")
    with cols2[1]:
        st.metric("p50", f"{mc.get('p50', 0):.3f}")
    with cols2[2]:
        st.metric("p95", f"{mc.get('p95', 0):.3f}")
    with cols2[3]:
        dist = mc.get("distribution_bins", [])
        if dist:
            # Show histogram
            df = pd.DataFrame(dist)
            st.bar_chart(df.set_index("bin_start")[["probability"]])

    st.caption(mc.get("interpretation", ""))


def _render_risk_detail(risk: dict) -> None:
    """Render risk detail."""
    st.markdown("### Risk Analysis")
    if not risk:
        st.caption("No risk detail available.")
        return

    cols = st.columns(3)
    with cols[0]:
        level = risk.get("level", "—")
        color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(level, "gray")
        st.metric("Risk Level", str(level))
        st.markdown(f"<span style='color:{color};font-size:2rem'>●</span>", unsafe_allow_html=True)
    with cols[1]:
        st.metric("Risk Score", f"{risk.get('score', 0):.3f}")
    with cols[2]:
        st.metric("Confidence", f"{risk.get('confidence', 0):.3f}")

    cols2 = st.columns(3)
    with cols2[0]:
        st.metric("Downside", f"{risk.get('downside', 0):.3f}")
    with cols2[1]:
        st.metric("Upside", f"{risk.get('upside', 0):.3f}")
    with cols2[2]:
        st.metric("Abstain", "Yes" if risk.get("abstain") else "No")


def _render_scenario_comparison(scenarios: list[dict]) -> None:
    """Render scenario comparison table."""
    st.markdown("### Scenario Comparison")
    if not scenarios:
        st.caption("No scenario comparison available.")
        return

    df = pd.DataFrame(scenarios)
    if df.empty:
        st.caption("Empty scenario list.")
        return

    # Show key columns
    key_cols = ["name", "oh", "nps", "probability", "confidence", "expected", "p05", "p50", "p95", "risk_severity"]
    display = [c for c in key_cols if c in df.columns]
    st.dataframe(df[display] if display else df, use_container_width=True)


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
    st.dataframe(df, use_container_width=True)

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
        st.metric("Strongest Positive", f"{sp.get('metric')}: {sp.get('change', 0):.2f}")
    if sn:
        st.metric("Strongest Negative", f"{sn.get('metric')}: {sn.get('change', 0):.2f}")

    analyses = trend.get("analyses", [])
    if analyses:
        df = pd.DataFrame(analyses)
        st.dataframe(df, use_container_width=True)


def _render_agreement_detail(agreement: dict) -> None:
    """Render agreement/conflicts."""
    st.markdown("### Agreement & Conflicts")
    if not agreement:
        st.caption("No agreement detail available.")
        return

    cols = st.columns(3)
    with cols[0]:
        st.metric("Agreement Score", f"{agreement.get('score', 0):.2f}")
    with cols[1]:
        st.metric("Category Consistency", f"{agreement.get('category_consistency', 0):.2f}")
    with cols[2]:
        st.metric("Conflict Count", agreement.get("conflict_count", 0))

    conflicts = agreement.get("conflicts", [])
    if conflicts:
        st.markdown("**Conflicts**")
        for c in conflicts[:5]:
            st.warning(str(c))


def _render_recommendations(recs: list[dict]) -> None:
    """Render top 3 recommendations."""
    st.markdown("### Top 3 Recommendations")
    if not recs:
        st.caption("No recommendations available.")
        return

    for r in recs[:3]:
        with st.expander(f"#{r.get('rank', '?')} — {r.get('action', 'Action')}", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                st.metric("Affected KPI", r.get("affected_kpi", "—"))
                st.metric("Direction", r.get("direction", "—"))
            with cols[1]:
                st.metric("Confidence", f"{r.get('confidence', 0):.2f}")
                st.metric("Risk", r.get("risk", "—"))
            eff = r.get("expected_effect", {})
            if eff:
                st.metric("Expected OH Gain", f"{eff.get('oh_gain', eff.get('oh_lift', eff.get('oh_improvement', '—')))}")
                st.metric("Expected NPS Gain", f"{eff.get('nps_gain', eff.get('nps_lift', '—'))}")
            if r.get("evidence"):
                st.caption("Evidence: " + "; ".join(str(e) for e in r["evidence"][:3]))


def _render_explanation(explanation: dict) -> None:
    """Render enhanced explanation."""
    st.markdown("### Decision Explanation")
    if not explanation:
        st.caption("No explanation available.")
        return

    st.markdown("**Current State**")
    cs = explanation.get("current_state", {})
    st.json(cs)

    st.markdown("**Forecast Outlook**")
    fo = explanation.get("forecast_summary", {})
    st.json(fo)

    st.markdown("**Why Selected**")
    ps = explanation.get("preferred_scenario", {})
    wp = explanation.get("why_preferred", {})
    if ps:
        st.write(f"Preferred: {ps.get('name')} (rank {ps.get('rank')}, score {ps.get('score'):.3f})")
    if wp:
        st.caption(wp.get("policy", ""))

    st.markdown("**Main Risk**")
    mr = explanation.get("main_risk", {})
    st.json(mr)

    st.markdown("**Uncertainty**")
    unc = explanation.get("uncertainty", {})
    st.json(unc)

    st.markdown("**What Would Change the Decision**")
    dc = explanation.get("decision_changers", {})
    st.json(dc)
