"""
core.decision_intelligence.v3.synthesis.decision_detail

Canonical ADIE V3 decision detail builder — Phase 16.

Assembles the full expanded detail (all 11 sections) from the existing
ProbabilisticDecisionPackage + Forecast AI outputs.

NEVER fabricates evidence. NEVER calls predictors. Never runs another
Monte Carlo — uses ONLY the single Bayesian + MC execution already
performed by ProbabilisticDecisionService.analyze.

All sections are optional (included only when evidence exists).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Optional


def _finite(value: Any) -> bool:
    """Check if value is a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _round(value: Any, digits: int = 4) -> Any:
    """Safely round a value to specified digits; return None if invalid."""
    v = _finite(value)
    if v is False:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def build_adie_detail(
    package: Mapping[str, Any],
    *,
    recommendation_output: Mapping[str, Any] | None = None,
    strategy_output: Mapping[str, Any] | None = None,
    trend_output: Mapping[str, Any] | None = None,
    sensitivity_output: Mapping[str, Any] | None = None,
    agreement: Mapping[str, Any] | None = None,
    targets: Mapping[str, Any] | None = None,
    observed: float | None = None,
    observed_metrics: Sequence[str] | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    """
    Build the canonical enriched ADIE detail package.

    Takes the probabilistic decision package (with bayesian_detail, monte_carlo_detail)
    and folds in Forecast AI outputs to produce the full expanded detail.

    Returns a dict with up to 11 sections:
        - recommendations (top 3 ranked)
        - forecast_summary
        - scenario_comparison
        - bayesian_detail
        - monte_carlo_detail
        - risk_detail
        - sensitivity_detail
        - trend_detail
        - agreement
        - explanation (enhanced)
        - best_scenario
    """
    # --- Extract package primitives ---
    scenarios = package.get("scenarios") or []
    best_scenario = scenarios[0] if scenarios else {}
    worst_scenario = scenarios[-1] if len(scenarios) > 1 else None

    semantics = package.get("semantics") or {}
    probability_of_target = semantics.get("probability_of_target") or {}

    bayesian_detail = package.get("bayesian_detail") or {}
    monte_carlo_detail = package.get("monte_carlo_detail") or {}
    explanation = package.get("explanation") or {}

    # --- 1. TOP RECOMMENDATIONS (up to 3 ranked) ---
    # Source: recommendation_output from Forecast AI, falling back to policy decision
    top_recommendations = _build_top_recommendations(
        recommendation_output=recommendation_output,
        policy_decision=package.get("decision"),
        scenarios=scenarios,
        targets=targets,
    )

    # --- 2. FORECAST SUMMARY ---
    forecast_summary = _build_forecast_summary(
        scenarios=scenarios,
        targets=targets,
        observed=observed,
        observed_metrics=observed_metrics,
        horizon=horizon,
    )

    # --- 3. SCENARIO COMPARISON (ranked list with full detail) ---
    scenario_comparison = _build_scenario_comparison(scenarios=scenarios)

    # --- 4. BAYESIAN DETAIL (from existing BayesianResult) ---
    bayesian = _build_bayesian_detail(
        bayesian_detail=bayesian_detail,
        best_scenario=best_scenario,
        probability_of_target=probability_of_target,
    )

    # --- 5. MONTE CARLO DETAIL (from existing MonteCarloResult) ---
    mc = _build_mc_detail(
        monte_carlo_detail=monte_carlo_detail,
    )

    # --- 6. RISK DETAIL ---
    risk_detail = _build_risk_detail(
        package=package,
        mc_detail=monte_carlo_detail,
    )

    # --- 7. SENSITIVITY DETAIL (from sensitivity_output) ---
    sensitivity_detail = _build_sensitivity_detail(sensitivity_output=sensitivity_output)

    # --- 8. TREND DETAIL (from trend_output) ---
    trend_detail = _build_trend_detail(trend_output=trend_output)

    # --- 9. AGREEMENT (from agreement dict) ---
    agreement_detail = _build_agreement_detail(agreement=agreement)

    # --- 10. ENHANCED EXPLANATION ---
    enhanced_explanation = _enhance_explanation(
        explanation=explanation,
        forecast_summary=forecast_summary,
        bayesian_detail=bayesian,
        mc_detail=mc,
    )

    return {
        "recommendations": top_recommendations,
        "forecast_summary": forecast_summary,
        "scenario_comparison": scenario_comparison,
        "bayesian_detail": bayesian,
        "monte_carlo_detail": mc,
        "risk_detail": risk_detail,
        "sensitivity_detail": sensitivity_detail,
        "trend_detail": trend_detail,
        "agreement": agreement_detail,
        "explanation": enhanced_explanation,
        "best_scenario": best_scenario,
    }


def _build_top_recommendations(
    recommendation_output: Mapping[str, Any] | None,
    policy_decision: Mapping[str, Any] | None,
    scenarios: list[dict[str, Any]],
    targets: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build up to 3 ranked recommendations."""
    recs = []
    targets = dict(targets or {})
    target_nps = targets.get("target_nps")
    target_oh = targets.get("target_operations_health")

    # Prefer Forecast AI recommendations when available
    if recommendation_output:
        data = recommendation_output.get("recommendations")
        if isinstance(data, list):
            # Sort by optimization_score descending (higher = better evidence)
            scored = []
            for r in data:
                if not isinstance(r, dict):
                    continue
                score = float(r.get("optimization_score", 0) or 0)
                priority_val = {"critical": 5, "high": 3, "medium": 2, "low": 1, "informational": 0}.get(
                    str(r.get("priority", "")).lower(), 0
                )
                scored.append((score, priority_val, r))
            scored.sort(key=lambda x: (-x[0], -x[1]))
            for _, _, r in scored[:3]:
                rank = len(recs) + 1
                prob = float(r.get("probability") or r.get("confidence") or 0.5)
                conf = float(r.get("confidence", 0.5))
                recs.append({
                    "action": r.get("title") or r.get("action") or f"Action {rank}",
                    "rank": rank,
                    "affected_kpi": r.get("target_kpi"),
                    "direction": r.get("direction"),
                    "expected_effect": {
                        "oh_gain": r.get("estimated_oh_gain"),
                        "nps_gain": r.get("estimated_nps_gain"),
                    },
                    "confidence": _round(conf),
                    "risk": _assess_risk(prob, conf) if _finite(prob) and _finite(conf) else "MEDIUM",
                    "evidence": r.get("reasoning") or r.get("actions", []),
                })
                if len(recs) >= 3:
                    break

    # Fallback: derive recommendations from top scenarios
    if len(recs) < 3:
        rank = len(recs) + 1
        best = scenarios[0] if scenarios else {}
        if best:
            nps = best.get("nps")
            oh = best.get("operations_health")
            delta = best.get("delta_oh")

            if target_nps is not None and nps is not None:
                if _finite(nps) and _finite(target_nps):
                    nps_conf = float(best.get("confidence", 0.5))
                    recs.append({
                        "action": "improve_nps" if nps < target_nps else "maintain_nps",
                        "rank": rank,
                        "affected_kpi": "nps",
                        "direction": "increase" if nps < target_nps else "maintain",
                        "expected_effect": {"nps_lift": target_nps - nps if _finite(nps) and _finite(target_nps) else None},
                        "confidence": _round(nps_conf),
                        "risk": _assess_risk(float(best.get("probability", 0.5)), nps_conf),
                        "evidence": [f"Current NPS: {_round(nps)}, Target: {target_nps}"],
                    })
                    rank += 1

            if target_oh is not None and oh is not None:
                if _finite(oh) and _finite(target_oh):
                    oh_conf = float(best.get("confidence", 0.5))
                    recs.append({
                        "action": "improve_oh" if oh < target_oh else "maintain_oh",
                        "rank": rank,
                        "affected_kpi": "operations_health",
                        "direction": "increase" if oh < target_oh else "maintain",
                        "expected_effect": {"oh_lift": target_oh - oh if _finite(oh) and _finite(target_oh) else None},
                        "confidence": _round(oh_conf),
                        "risk": _assess_risk(float(best.get("probability", 0.5)), oh_conf),
                        "evidence": [f"Current OH: {_round(oh)}, Target: {target_oh}"],
                    })
                    rank += 1

        # If fewer than 3, add generic action if there's improvement potential
        if len(recs) < 3 and delta is not None and _finite(delta) and delta > 0:
            opt_conf = float(best.get("confidence", 0.5))
            recs.append({
                "action": "pursue_optimization",
                "rank": rank,
                "affected_kpi": "hybrid",
                "direction": "improve",
                "expected_effect": {"oh_improvement": _round(delta)},
                "confidence": _round(opt_conf),
                "risk": _assess_risk(float(best.get("probability", 0.5)), opt_conf),
                "evidence": [f"Delta OH from optimization: {_round(delta)}"],
            })

    return recs[:3]


def _assess_risk(probability: float, confidence: float) -> str:
    """Assess risk level from probability and confidence."""
    prob = _round(probability, 2)
    conf = _round(confidence, 2)
    if prob is None or conf is None:
        return "MEDIUM"
    if prob < 0.5 or conf < 0.5:
        return "HIGH"
    if prob < 0.7 or conf < 0.7:
        return "MEDIUM"
    return "LOW"


def _build_forecast_summary(
    scenarios: list[dict[str, Any]],
    targets: Mapping[str, Any] | None,
    observed: float | None,
    observed_metrics: Sequence[str] | None,
    horizon: int | None,
) -> dict[str, Any]:
    """Build forecast summary with OH/NPS ranges and per-day table."""
    if not scenarios:
        return {"note": "No forecast scenarios available"}

    oh_values = [s.get("operations_health") for s in scenarios if _finite(s.get("operations_health"))]
    nps_values = [s.get("nps") for s in scenarios if _finite(s.get("nps"))]

    oh_min = min(oh_values) if oh_values else None
    oh_max = max(oh_values) if oh_values else None
    oh_expected = scenarios[0].get("operations_health") if scenarios else None

    nps_min = min(nps_values) if nps_values else None
    nps_max = max(nps_values) if nps_values else None
    nps_expected = scenarios[0].get("nps") if scenarios else None

    # Best/worst day
    best_day = scenarios[0] if scenarios else {}
    worst_day = scenarios[-1] if len(scenarios) > 1 else best_day
    expected_day = best_day

    # Per-day forecast table
    daily_table = []
    for i, s in enumerate(scenarios):
        day_entry = {
            "day": i + 1,
            "oh": _round(s.get("operations_health")),
            "nps": _round(s.get("nps")),
            "confidence": _round(s.get("confidence")),
            "risk": _round(s.get("risk_severity")),
            "_predicted": _round(s.get("expected")),
        }
        # Add NPS distribution info if present
        if "nps_p05" in s:
            day_entry["nps_p05"] = _round(s["nps_p05"])
            day_entry["nps_p95"] = _round(s.get("nps_p95"))
        if "expected_nps" in s:
            day_entry["expected_nps"] = _round(s.get("expected_nps"))
        if "bayesian_score_distribution" in s:
            day_entry["nps_distribution"] = s["bayesian_score_distribution"]
        daily_table.append(day_entry)

    return {
        "oh_range": {"min": _round(oh_min), "max": _round(oh_max), "expected": _round(oh_expected)},
        "nps_range": {"min": _round(nps_min), "max": _round(nps_max), "expected": _round(nps_expected)},
        "best_day": {"day_index": 1, "oh": _round(best_day.get("operations_health")), "nps": _round(best_day.get("nps"))},
        "worst_day": {"day_index": len(scenarios), "oh": _round(worst_day.get("operations_health")), "nps": _round(worst_day.get("nps"))},
        "expected_day": {"day_index": 1, "oh": _round(expected_day.get("operations_health")), "nps": _round(expected_day.get("nps"))},
        "per_day_table": daily_table,
        "scenario_count": len(scenarios),
        "horizon_days": horizon,
    }


def _build_scenario_comparison(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ranked scenario comparison with full detail."""
    result = []
    for s in scenarios:
        entry = {
            "name": s.get("name"),
            "oh": _round(s.get("operations_health")),
            "nps": _round(s.get("nps")),
            "probability": _round(s.get("probability")),
            "confidence": _round(s.get("confidence")),
            "expected": _round(s.get("expected")),
            "p05": _round(s.get("p05")),
            "p50": _round(s.get("p50")),
            "p95": _round(s.get("p95")),
            "risk_severity": _round(s.get("risk_severity")),
        }
        # Include _predicted if present
        if "_predicted" in s:
            entry["_predicted"] = s["_predicted"]
        result.append(entry)
    return result


def _build_bayesian_detail(
    bayesian_detail: dict[str, Any],
    best_scenario: dict[str, Any],
    probability_of_target: dict[str, Any],
) -> dict[str, Any]:
    """Build Bayesian detail from existing result + best scenario distribution."""
    result = {
        "decision_probability": bayesian_detail.get("probability"),
        "confidence": bayesian_detail.get("confidence"),
        "posterior_ranges": {
            "mean": bayesian_detail.get("posterior_mean"),
            "std": bayesian_detail.get("posterior_std"),
            "credible_interval": {
                "lower": bayesian_detail.get("credible_interval_lower"),
                "upper": bayesian_detail.get("credible_interval_upper"),
                "level": bayesian_detail.get("credible_level"),
            },
        },
    }

    # Target probabilities where available
    target_probs = {}
    if probability_of_target:
        interpretation = probability_of_target.get("interpretation")
        if "nps" in probability_of_target:
            target_probs["nps"] = {
                "expected_nps": probability_of_target["nps"].get("expected_nps"),
                "probability_promoter": probability_of_target["nps"].get("probability_promoter_score"),
                "target": probability_of_target["nps"].get("target"),
            }
        if "operations_health" in probability_of_target:
            target_probs["operations_health"] = probability_of_target["operations_health"]
    result["probability_of_target"] = target_probs or None

    # NPS 0-10 distribution from best scenario
    nps_dist = None
    if best_scenario:
        nps_dist = best_scenario.get("bayesian_score_distribution") or best_scenario.get("nps_distribution")
    if nps_dist:
        result["nps_0_10_distribution"] = nps_dist
        # Expected NPS business value (scale -100..100)
        try:
            # import from bayesian_inference
            from core.decision_intelligence.v3.bayesian import inference as bi
            result["nps_expected_business"] = bi.expected_nps_business(nps_dist) if nps_dist else None
            result["nps_promoter_probability"] = bi.promoter_probability(nps_dist) if nps_dist else None
        except Exception:
            result["nps_expected_business"] = None
            result["nps_promoter_probability"] = None

    # Interpretation notes
    result["interpretation"] = (
        "Probability represents the Beta-Bernoulli posterior mean over normalized "
        "health-score observations. Confidence = 1 - normalized posterior std. "
        "Target probabilities are derived from existing forecast distributions, not "
        "recomputed Monte Carlo."
    )
    return result


def _build_mc_detail(monte_carlo_detail: dict[str, Any]) -> dict[str, Any]:
    """Build Monte Carlo detail from existing result."""
    distribution_summary = monte_carlo_detail.get("distribution_summary", {})
    success_pct = None
    failure_pct = None
    if monte_carlo_detail.get("success_count") is not None:
        total = monte_carlo_detail.get("success_count", 0) + monte_carlo_detail.get("failure_count", 0)
        if total > 0:
            success_pct = round(monte_carlo_detail.get("success_count", 0) / total * 100, 2)
            failure_pct = round(monte_carlo_detail.get("failure_count", 0) / total * 100, 2)

    return {
        "total_samples": distribution_summary.get("samples"),
        "success_count": monte_carlo_detail.get("success_count"),
        "failure_count": monte_carlo_detail.get("failure_count"),
        "success_percentage": success_pct,
        "failure_percentage": failure_pct,
        "expected_value": distribution_summary.get("mean"),
        "p05": distribution_summary.get("p05"),
        "p50": distribution_summary.get("p50"),
        "p95": distribution_summary.get("p95"),
        "distribution_bins": monte_carlo_detail.get("distribution", []),
        "uncertainty": distribution_summary.get("uncertainty"),
        "interpretation": (
            "Monte Carlo samples from a single simulation. Values > 0 are "
            "counted as success; all samples derive from a normal distribution "
            "centered on the expected value with scale = uncertainty."
        ),
    }


def _build_risk_detail(
    package: Mapping[str, Any],
    mc_detail: dict[str, Any],
) -> dict[str, Any]:
    """Build risk detail from package, using real thresholds from policy constants."""
    from core.decision_intelligence.v3.policy import constants as pc

    downside = package.get("downside")
    if downside is None and mc_detail:
        downside = mc_detail.get("distribution_summary", {}).get("p05")
    upside = package.get("upside")
    if upside is None and mc_detail:
        upside = mc_detail.get("distribution_summary", {}).get("p95")

    return {
        "level": package.get("risk"),
        "score": _round(package.get("risk_score")),
        "confidence": _round(package.get("confidence")),
        "downside": _round(downside),
        "upside": _round(upside),
        "abstain": package.get("abstain"),
        "drivers": {
            "high_risk": f"confidence < {pc.RISK_THRESHOLDS.get('high_confidence', 0.35):.2f} or probability < {pc.RISK_THRESHOLDS.get('high_probability', 0.35):.2f}",
            "medium_risk": f"confidence < {pc.RISK_THRESHOLDS.get('medium_confidence', 0.60):.2f} or probability < {pc.RISK_THRESHOLDS.get('medium_probability', 0.60):.2f}",
        },
        "threshold_policy": pc.DECISION_POLICY if hasattr(pc, 'DECISION_POLICY') else {},
    }


def _build_sensitivity_detail(sensitivity_output: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build sensitivity detail from Forecast AI sensitivity output."""
    if not sensitivity_output:
        return {"note": "Sensitivity analysis not available (forecast not run)"}

    analyses = sensitivity_output.get("analyses") or []
    ranking = sensitivity_output.get("ranking") or []

    # Extract the 5 key sensitivities: quality, competency, attendance, release, transfer
    sensitivity_metrics = ["quality", "competency", "attendance", "release", "transfer"]
    details = []

    for metric in sensitivity_metrics:
        # Find analysis for this metric
        analysis = None
        for a in analyses:
            if isinstance(a, dict) and a.get("metric", "").lower() == metric.lower():
                analysis = a
                break

        if analysis:
            oh_change = _round(analysis.get("oh_change") or analysis.get("operations_health_change"))
            nps_change = _round(analysis.get("nps_change") or analysis.get("modified_nps", {}).get("nps_change") or 0)
            sens_oh = _round(analysis.get("sensitivity_oh"))
            sens_nps = _round(analysis.get("sensitivity_nps"))
            direction = "increase" if (oh_change is not None and oh_change > 0) else ("decrease" if oh_change is not None else "unknown")
            rel_impact = "high" if (sens_oh is not None and abs(sens_oh) > 1.0) else ("medium" if sens_oh is not None else "low")
            details.append({
                "metric": metric,
                "oh_change": oh_change,
                "nps_change": nps_change,
                "direction": direction,
                "relative_impact": rel_impact,
                "interpretation": f"Increasing {metric} by 1pp {'improves' if direction == 'increase' else 'reduces'} OH by {sens_oh}pp" if sens_oh else "Unknown impact",
            })

    return {
        "metrics": details,
        "ranking": [m.get("metric") for m in ranking[:5]] if ranking else [],
    }


def _build_trend_detail(trend_output: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build trend detail from Forecast AI trend output."""
    if not trend_output:
        return {"note": "Trend analysis not available"}

    analyses = trend_output.get("analyses") or []

    improvements = []
    declines = []
    for a in analyses:
        direction = a.get("direction", "")
        if "Increase" in direction or "Strong" in direction:
            improvements.append(a)
        if "Decrease" in direction:
            declines.append(a)

    strongest_positive = improvements[0] if improvements else None
    strongest_negative = declines[0] if declines else None

    return {
        "direction": "stable" if not improvements and not declines else ("improving" if improvements else "declining"),
        "slope_change": [a.get("absolute_change") for a in analyses],
        "strongest_positive": {
            "metric": strongest_positive.get("metric") if strongest_positive else None,
            "change": _round(strongest_positive.get("absolute_change")) if strongest_positive else None,
            "direction": strongest_positive.get("direction") if strongest_positive else None,
        } if strongest_positive else None,
        "strongest_negative": {
            "metric": strongest_negative.get("metric") if strongest_negative else None,
            "change": _round(strongest_negative.get("absolute_change")) if strongest_negative else None,
            "direction": strongest_negative.get("direction") if strongest_negative else None,
        } if strongest_negative else None,
        "analyses": [{
            "metric": a.get("metric"),
            "direction": a.get("direction"),
            "change": _round(a.get("absolute_change")),
            "pct_change": _round(a.get("percent_change")),
        } for a in analyses],
    }


def _build_agreement_detail(agreement: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build agreement detail from Forecast AI agreement output."""
    if not agreement:
        return {"note": "Agreement analysis not available"}

    conflicts = agreement.get("conflicts") or []
    return {
        "score": _round(agreement.get("score")),
        "category_consistency": agreement.get("category_consistency"),
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:5],  # Top 5
    }


def _enhance_explanation(
    explanation: dict[str, Any],
    forecast_summary: dict[str, Any],
    bayesian_detail: dict[str, Any],
    mc_detail: dict[str, Any],
) -> dict[str, Any]:
    """Enhance the existing explanation with additional detail."""
    enhanced = dict(explanation) if explanation else {}

    enhanced["bayesian"] = bayesian_detail
    enhanced["monte_carlo"] = mc_detail
    enhanced["forecast_summary"] = forecast_summary

    return enhanced


__all__ = [
    "build_adie_detail",
]
