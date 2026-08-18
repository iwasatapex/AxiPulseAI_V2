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


def canonical_recommendation_list(
    recommendation_output: Mapping[str, Any] | None,
) -> list[Any]:
    """Extract the canonical recommendation evidence list.

    The canonical recommendation-evidence representation is::

        {"status": "success", "success": bool,
         "recommendations": [ {...rec...} ],   # flat list
         "evidence_count": N,
         "final_recommendation_count": N,
         "diagnostics": {...}}

    Every consumer (decision_evidence_sufficient, _build_top_recommendations,
    _compute_agreement) reads from the SAME ``recommendations`` key. This helper
    returns that canonical list regardless of whether the input is the canonical
    flat shape or a legacy NESTED shape::

        {"recommendations": {"success": bool, "recommendations": [...], ...}}

    It never fabricates evidence — it only unwraps the same genuine list. An
    empty/missing representation yields an empty list.
    """
    if not recommendation_output:
        return []
    recs = recommendation_output.get("recommendations")
    if isinstance(recs, list):
        return recs
    # Legacy nested block: {"recommendations": {"success":..., "recommendations":[...]}}
    if isinstance(recs, dict):
        inner = recs.get("recommendations")
        if isinstance(inner, list):
            return inner
    return []


# Canonical decision evidence gate statuses.
DECISION_STATUS_AVAILABLE = "available"
DECISION_STATUS_INSUFFICIENT = "insufficient_evidence"


def decision_evidence_sufficient(
    recommendation_output: Mapping[str, Any] | None,
    agreement: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Decide whether the canonical decision/recommendation evidence gate is met.

    The canonical decision/recommendation/risk output requires BOTH genuine
    recommendation evidence (a non-empty recommendation list produced by the
    Forecast AI engines) AND a computed agreement/consistency (a real score and
    category_consistency). Missing evidence is NEVER treated as neutral or zero
    — it is an explicit ``insufficient_evidence`` gate.

    Returns ``(sufficient, reason)``.
    """
    if recommendation_output is None:
        return False, "No recommendation evidence available"

    data = canonical_recommendation_list(recommendation_output)
    status = str(recommendation_output.get("status", "success")).lower()
    if not data:
        if status in ("skipped", "error", "failed"):
            return False, "Recommendation engine did not produce recommendations"
        return False, "Recommendation evidence is empty"

    if agreement is None:
        return False, "No recommendation evidence to compute agreement/consistency."

    score = agreement.get("score")
    consistency = agreement.get("category_consistency")
    if score is None or consistency is None:
        return (
            False,
            "Agreement evidence is incomplete (score or consistency missing).",
        )

    return True, "Sufficient recommendation and agreement evidence present"


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

    # --- Evidence gate -------------------------------------------------------
    # The canonical decision/recommendation/risk output is only produced when
    # genuine recommendation AND agreement evidence are present. When that gate
    # is not met the decision is explicitly withheld (ABSTAIN) rather than
    # presenting a normal LOW/MEDIUM/HIGH decision on insufficient evidence.
    evidence_sufficient, evidence_reason = decision_evidence_sufficient(
        recommendation_output, agreement
    )

    # --- 1. TOP RECOMMENDATIONS (up to 3 ranked) ---
    # Source: recommendation_output from Forecast AI, falling back to policy decision
    top_recommendations = _build_top_recommendations(
        recommendation_output=recommendation_output,
        policy_decision=package.get("decision"),
        scenarios=scenarios,
        targets=targets,
        evidence_sufficient=evidence_sufficient,
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
        targets=targets,
    )

    # --- 6. RISK DETAIL ---
    risk_detail = _build_risk_detail(
        package=package,
        mc_detail=monte_carlo_detail,
        evidence_sufficient=evidence_sufficient,
        evidence_reason=evidence_reason,
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
        evidence_sufficient=evidence_sufficient,
        evidence_reason=evidence_reason,
        preferred_name=(best_scenario or {}).get("name"),
    )

    decision_status = (
        DECISION_STATUS_AVAILABLE if evidence_sufficient else DECISION_STATUS_INSUFFICIENT
    )
    return {
        "decision_status": decision_status,
        "recommendation_status": decision_status,
        "decision_evidence": {
            "sufficient": evidence_sufficient,
            "reason": evidence_reason,
            "recommendation_status": decision_status,
            "agreement_status": agreement_detail.get("status", DECISION_STATUS_INSUFFICIENT),
        },
        "scenario_ranking": {
            "status": DECISION_STATUS_AVAILABLE if evidence_sufficient else "forecast_ranking_only",
            "label": (
                "Scenario ranking"
                if evidence_sufficient
                else "Forecast ranking only — insufficient decision evidence"
            ),
            "actionable": evidence_sufficient,
        },
        "forecast_preference": {
            "name": (best_scenario or {}).get("name"),
            "rank": (best_scenario or {}).get("rank"),
            "score": _round((best_scenario or {}).get("score")),
            "actionable": evidence_sufficient,
            "kind": "forecast_preference",
        },
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
    *,
    evidence_sufficient: bool = True,
) -> list[dict[str, Any]]:
    """Build up to 3 ranked recommendations — never fabricating.

    Priority:
      1. Recommendations from the Forecast AI engine (ranked by evidence).
      2. Scenario/target fallback: KPI-specific recs derived from forecast
         scenario values vs. explicit targets (real numbers, genuine evidence).
      3. If the engine was explicitly skipped (status=skipped/error/failed
         with empty recommendations) no fallback is attempted.

    Never adds generic ``pursue_optimization`` just to reach 3.

    Forecast AI recommendations are preserved as advisory evidence even when
    the canonical decision evidence gate is insufficient. The evidence gate
    controls whether a canonical decision is actionable; it does not erase
    genuine upstream recommendation evidence.
    """
    recs = []
    targets = dict(targets or {})
    target_nps = targets.get("target_nps")
    target_oh = targets.get("target_operations_health") or targets.get("target_oh")

    # 1. Forecast AI recommendation list
    source_list = None
    engine_skipped = False
    if recommendation_output:
        data = canonical_recommendation_list(recommendation_output)
        status = str(recommendation_output.get("status", "success")).lower()
        if data:
            source_list = data
        elif status in ("skipped", "error", "failed"):
            engine_skipped = True  # explicit empty result — always respect

    if source_list:
        # Rank by optimization_score (higher = stronger evidence)
        scored = []
        for r in source_list:
            if not isinstance(r, dict):
                continue
            score = float(r.get("optimization_score", 0) or 0)
            priority_map = {"critical": 5, "high": 3, "medium": 2, "low": 1, "informational": 0}
            prio = priority_map.get(str(r.get("priority", "")).lower(), 0)
            scored.append((score, prio, r))
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

    # 2. Scenario/target fallback — only when the engine produced no recs
    #    AND the engine was not explicitly skipped/errored.
    if not recs and not engine_skipped:
        best = scenarios[0] if scenarios else {}
        if best:
            nps = best.get("nps")
            oh = best.get("operations_health")
            rank = 1

            if target_nps is not None and nps is not None:
                if _finite(nps) and _finite(float(target_nps)):
                    nps_conf = float(best.get("confidence", 0.5))
                    recs.append({
                        "action": "improve_nps" if nps < float(target_nps) else "maintain_nps",
                        "rank": rank,
                        "affected_kpi": "nps",
                        "direction": "increase" if nps < float(target_nps) else "maintain",
                        "expected_effect": {"nps_lift": float(target_nps) - nps},
                        "confidence": _round(nps_conf),
                        "risk": _assess_risk(float(best.get("probability", 0.5)), nps_conf),
                        "evidence": [f"Current NPS: {_round(nps)}, Target: {target_nps}"],
                    })
                    rank += 1

            if target_oh is not None and oh is not None:
                if _finite(oh) and _finite(float(target_oh)):
                    oh_conf = float(best.get("confidence", 0.5))
                    recs.append({
                        "action": "improve_oh" if oh < float(target_oh) else "maintain_oh",
                        "rank": rank,
                        "affected_kpi": "operations_health",
                        "direction": "increase" if oh < float(target_oh) else "maintain",
                        "expected_effect": {"oh_lift": float(target_oh) - oh},
                        "confidence": _round(oh_conf),
                        "risk": _assess_risk(float(best.get("probability", 0.5)), oh_conf),
                        "evidence": [f"Current OH: {_round(oh)}, Target: {target_oh}"],
                    })

    return recs[:3]


def _assess_risk(probability: float, confidence: float) -> str:
    """Canonical risk level for recommendation labels.

    Delegates to ``UncertaintyRiskEngine.classify_level`` so every ADIE
    surface (Risk Analysis, decision detail, recommendations, API) shares the
    exact same risk semantics — there is only ONE risk model. Missing /
    non-finite inputs map to the documented neutral "MEDIUM" label (no data),
    never to alternate thresholds.
    """
    from core.decision_intelligence.v3.risk.uncertainty import UncertaintyRiskEngine

    if not (_finite(probability) and _finite(confidence)):
        return "MEDIUM"
    return UncertaintyRiskEngine.classify_level(
        probability=float(probability),
        confidence=float(confidence),
    )


def _day_summary(s: dict[str, Any], index: int) -> dict[str, Any]:
    """Present a ranked forecast scenario as a best/worst/expected day entry.

    ``index`` is the 0-based position in the pre-ranked scenario list; the
    actual ADIE rank is ``s['rank']`` when present (the source of truth).
    ``day_index`` mirrors the rank so it is never force-pinned to day 1.
    """
    rank = s.get("rank", index + 1)
    return {
        "day_index": rank,
        "rank": rank,
        "name": s.get("name"),
        "score": _round(s.get("score")),
        "oh": _round(s.get("operations_health")),
        "nps": _round(s.get("nps")),
        "factors": (s.get("evidence") or {}).get("available", []),
    }


def _build_forecast_summary(
    scenarios: list[dict[str, Any]],
    targets: Mapping[str, Any] | None,
    observed: float | None,
    observed_metrics: Sequence[str] | None,
    horizon: int | None,
) -> dict[str, Any]:
    """Build forecast summary with OH/NPS ranges and per-day table.

    Naming (Phase 16 fix): ``oh_range`` / ``nps_range`` are the POINT forecast
    ranges (min/max across the ranked forecast days, with the expected value =
    the top-ranked scenario). They are labelled as such and are never presented
    as a confidence interval. When p05/p95 are present on the scenarios they
    are surfaced separately as the probabilistic interval.
    """
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

    # Forecast character (honest labeling, never fabrication): detect a flat
    # point forecast (identical OH/NPS across days) plus horizon confidence
    # decay, and surface it explicitly so the API/UI never presents the repeated
    # values as a genuinely day-specific forecast.
    _identical = lambda vals: len(vals) > 1 and all(abs(v - vals[0]) < 1e-9 for v in vals)
    flat_oh = _identical(oh_values)
    flat_nps = _identical(nps_values)
    confidences = [
        c for c in (s.get("confidence") for s in scenarios)
        if isinstance(c, (int, float)) and _finite(c)
    ]
    declining_conf = (
        len(confidences) > 1
        and all(
            confidences[i] <= confidences[i - 1] + 1e-9
            for i in range(1, len(confidences))
        )
    )
    forecast_character = {
        "type": "point_forecast",
        "flat_oh": flat_oh,
        "flat_nps": flat_nps,
        "horizon_confidence_decay": bool(declining_conf),
        "note": (
            "Per-day OH/NPS are point forecasts. When no day-specific scenario "
            "variance is supplied they are identical across days; uncertainty is "
            "expressed as horizon confidence decay, not as distinct day forecasts."
            if (flat_oh and flat_nps)
            else "Per-day OH/NPS vary across the forecast horizon."
        ),
    }

    # Best/worst/expected day = the actual pre-ranked scenario order.
    best_day = _day_summary(scenarios[0], 0)
    worst_day = _day_summary(scenarios[-1], len(scenarios) - 1) if len(scenarios) > 1 else best_day
    expected_day = best_day

    # Probabilistic interval (p05/p95) from the ranked days where available.
    oh_p05s = [v for v in (_round(s.get("p05")) for s in scenarios) if v is not None]
    oh_p95s = [v for v in (_round(s.get("p95")) for s in scenarios) if v is not None]
    oh_p05 = min(oh_p05s) if oh_p05s else None
    oh_p95 = max(oh_p95s) if oh_p95s else None

    # Per-day forecast table
    daily_table = []
    for i, s in enumerate(scenarios):
        day_entry = {
            "day": i + 1,
            "oh": _round(s.get("operations_health")),
            "nps": _round(s.get("nps")),
            "confidence": _round(s.get("confidence")),
            "risk": _round(s.get("risk_severity")),
            # Real boolean from source (never derived from 'expected').
            "_predicted": s.get("_predicted"),
        }
        if "score_p05" in s:
            day_entry["score_p05"] = _round(s["score_p05"])
            day_entry["score_p95"] = _round(s.get("score_p95"))
        if "expected_score" in s:
            day_entry["expected_score"] = _round(s.get("expected_score"))
        if "nps_p05" in s:
            day_entry["nps_p05"] = _round(s["nps_p05"])
            day_entry["nps_p95"] = _round(s.get("nps_p95"))
        if "bayesian_score_distribution" in s:
            day_entry["nps_distribution"] = s["bayesian_score_distribution"]
        daily_table.append(day_entry)

    return {
        "character": forecast_character,
        "oh_range": {
            "type": "point_forecast_range",
            "label": "Point forecast range (min/max across forecast days)",
            "min": _round(oh_min),
            "max": _round(oh_max),
            "expected": _round(oh_expected),
        },
        "nps_range": {
            "type": "point_forecast_range",
            "label": "Point forecast range (min/max across forecast days)",
            "min": _round(nps_min),
            "max": _round(nps_max),
            "expected": _round(nps_expected),
        },
        "probabilistic_interval": {
            "label": "Probabilistic interval (p05/p95 of the single Monte Carlo)",
            "oh_p05": oh_p05,
            "oh_p95": oh_p95,
        },
        "best_day": best_day,
        "worst_day": worst_day,
        "expected_day": expected_day,
        "per_day_table": daily_table,
        "scenario_count": len(scenarios),
        "horizon_days": horizon,
    }


def _build_scenario_comparison(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ranked scenario comparison with full detail (actual ADIE rank).

    Only keys that carry a real value are included — forecast-day scenarios
    are not fabricated with aggregate-only stats, so probability/expected/p05
    /p50/p95 are omitted (not ``None``) when the scenario lacks them. This
    keeps the payload compact and lets views hide all-empty columns.
    """
    result = []
    for i, s in enumerate(scenarios):
        entry = {
            "name": s.get("name"),
            "rank": s.get("rank", i + 1),
            "_predicted": s.get("_predicted"),
        }
        for key, source in (
            ("score", "score"),
            ("oh", "operations_health"),
            ("nps", "nps"),
            ("probability", "probability"),
            ("confidence", "confidence"),
            ("expected", "expected"),
            ("p05", "p05"),
            ("p50", "p50"),
            ("p95", "p95"),
            ("risk_severity", "risk_severity"),
        ):
            value = _round(s.get(source))
            if value is not None:
                entry[key] = value
        evidence = s.get("evidence") or {}
        if evidence:
            entry["ranking_factors"] = evidence.get("available", [])
            entry["ranking_policy"] = evidence.get("policy")
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
            # Mapping note: the source semantics field is
            # ``probability_of_target.nps.probability_promoter_score``
            # (P(a survey score is a promoter)); it is surfaced here as
            # ``probability_promoter`` for display continuity. Same value,
            # renamed — the source key is never silently dropped.
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


def _build_mc_detail(
    monte_carlo_detail: dict[str, Any],
    targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Monte Carlo detail from the EXISTING single result.

    Success semantics (Phase 16 fix):
      - When a target is defined, success = the simulated OH outcome actually
        satisfies the target (P(OH >= target_oh)). That probability is derived
        from the one existing Monte Carlo distribution (mean + p05/p95 -> std)
        via a normal CDF — NO second simulation is run.
      - When no target is defined, the success rate is explicitly UNAVAILABLE
        rather than inventing a "sample > 0" success definition.
    """
    from core.probabilistic.adapter import UniversalProbabilisticAdapter

    distribution_summary = monte_carlo_detail.get("distribution_summary", {}) or {}
    samples = distribution_summary.get("samples")
    mean = distribution_summary.get("mean")
    p05 = distribution_summary.get("p05")
    p95 = distribution_summary.get("p95")

    success_count = failure_count = success_pct = failure_pct = None
    success_definition = "unavailable"
    interpretation = (
        "Monte Carlo success rate unavailable: no target defined. Samples, "
        "percentiles and distribution bins are from the single existing "
        "Monte Carlo execution."
    )

    targets = dict(targets or {})
    target_oh = targets.get("target_operations_health") or targets.get("target_oh")
    if (
        target_oh is not None
        and _finite(target_oh)
        and _finite(mean)
        and _finite(p05)
        and _finite(p95)
    ):
        # Derive the single-execution normal std from its p05/p95, then the
        # probability that the simulated outcome satisfies the target.
        std = max((float(p95) - float(p05)) / 3.289707253, 0.0)
        target_ratio = float(target_oh) / 100.0
        prob = UniversalProbabilisticAdapter._probability_at_or_above(
            float(mean),
            target_ratio,
            std,
        )
        prob = max(0.0, min(1.0, float(prob)))
        if samples is not None and _finite(samples):
            n = int(round(float(samples)))
            success_count = int(round(prob * n))
            failure_count = n - success_count
            success_pct = round(prob * 100.0, 2)
            failure_pct = round((1.0 - prob) * 100.0, 2)
        success_definition = f"simulated OH outcome >= target_oh={float(target_oh)}"
        interpretation = (
            f"Monte Carlo success defined as the simulated OH outcome satisfying "
            f"the target (P(OH >= {float(target_oh)}) = {prob:.4f}), derived from "
            "the single existing Monte Carlo distribution. Counts are computed from "
            "that one execution."
        )

    return {
        "total_samples": _round(samples, 0),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_percentage": success_pct,
        "failure_percentage": failure_pct,
        "success_definition": success_definition,
        "expected_value": _round(mean),
        "p05": _round(p05),
        "p50": _round(distribution_summary.get("p50")),
        "p95": _round(p95),
        "distribution_bins": monte_carlo_detail.get("distribution", []) or [],
        "uncertainty": _round(distribution_summary.get("uncertainty")),
        "interpretation": interpretation,
    }


def _build_risk_detail(
    package: Mapping[str, Any],
    mc_detail: dict[str, Any],
    *,
    evidence_sufficient: bool = True,
    evidence_reason: str | None = None,
) -> dict[str, Any]:
    """Build risk detail from package, using real thresholds from policy constants.

    When the decision evidence gate is insufficient the canonical risk decision
    is ABSTAIN (``abstain=True``); the raw probability/confidence-derived score
    and tails are retained ONLY as diagnostic metadata and never presented as a
    valid canonical risk decision.
    """
    from core.decision_intelligence.v3.policy import constants as pc

    downside = package.get("downside")
    if downside is None and mc_detail:
        downside = mc_detail.get("distribution_summary", {}).get("p05")
    upside = package.get("upside")
    if upside is None and mc_detail:
        upside = mc_detail.get("distribution_summary", {}).get("p95")

    drivers = {
        "high_risk": f"confidence < {pc.RISK_THRESHOLDS.get('high_confidence', 0.35):.2f} or probability < {pc.RISK_THRESHOLDS.get('high_probability', 0.35):.2f}",
        "medium_risk": f"confidence < {pc.RISK_THRESHOLDS.get('medium_confidence', 0.60):.2f} or probability < {pc.RISK_THRESHOLDS.get('medium_probability', 0.60):.2f}",
    }

    if not evidence_sufficient:
        # Canonical decision surface is ABSTAIN. Raw probabilistic risk is
        # retained only under ``raw`` for diagnostics.
        raw_level = package.get("risk")
        return {
            "level": "ABSTAIN",
            "abstain": True,
            "status": DECISION_STATUS_INSUFFICIENT,
            "reason": evidence_reason or "Decision evidence is insufficient",
            "canonical": {
                "level": "ABSTAIN",
                "abstain": True,
                "status": DECISION_STATUS_INSUFFICIENT,
            },
            "raw": {
                "level": raw_level,
                "score": _round(package.get("risk_score")),
                "confidence": _round(package.get("confidence")),
                "downside": _round(downside),
                "upside": _round(upside),
            },
            "score": None,
            "confidence": None,
            "downside": None,
            "upside": None,
            "drivers": drivers,
            "threshold_policy": pc.DECISION_POLICY if hasattr(pc, "DECISION_POLICY") else {},
        }

    return {
        "level": package.get("risk"),
        "score": _round(package.get("risk_score")),
        "confidence": _round(package.get("confidence")),
        "downside": _round(downside),
        "upside": _round(upside),
        "abstain": package.get("abstain"),
        "drivers": drivers,
        "threshold_policy": pc.DECISION_POLICY if hasattr(pc, "DECISION_POLICY") else {},
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
            # Forecast AI serializes SensitivityAnalysis dataclasses via asdict(),
            # so the canonical keys are sensitivity_score_oh / sensitivity_score_nps.
            # The bare sensitivity_oh / sensitivity_nps keys are accepted for
            # backward compatibility with older dict-shaped inputs.
            sens_oh = _round(
                analysis.get("sensitivity_score_oh")
                if analysis.get("sensitivity_score_oh") is not None
                else analysis.get("sensitivity_oh")
            )
            sens_nps = _round(
                analysis.get("sensitivity_score_nps")
                if analysis.get("sensitivity_score_nps") is not None
                else analysis.get("sensitivity_nps")
            )
            canonical_direction = (
                "decrease"
                if metric.lower() == "transfer"
                else "increase"
            )

            # Raw model derivative: dOH / dKPI.
            raw_sensitivity_oh = sens_oh

            # Direction-aware operational sensitivity. This is a derived
            # interpretation only; raw_sensitivity_oh is never modified.
            improvement_sensitivity_oh = (
                None
                if raw_sensitivity_oh is None
                else (
                    -raw_sensitivity_oh
                    if canonical_direction == "decrease"
                    else raw_sensitivity_oh
                )
            )

            model_conflict = (
                improvement_sensitivity_oh is not None
                and improvement_sensitivity_oh < 0.0
            )

            if sens_oh is None:
                rel_impact = "low"
            else:
                # Magnitude-aligned with the canonical SENSITIVITY_THRESHOLDS
                # (very_high 1.0 / high 0.5 / medium 0.2 / low 0.05).
                abs_sens = abs(sens_oh)
                if abs_sens > 0.5:
                    rel_impact = "high"
                elif abs_sens > 0.2:
                    rel_impact = "medium"
                elif abs_sens > 0.05:
                    rel_impact = "low"
                else:
                    rel_impact = "negligible"

            if improvement_sensitivity_oh is None:
                interpretation = "Unknown impact"
            elif model_conflict:
                interpretation = (
                    f"Model conflict: the operationally beneficial direction "
                    f"for {metric} is {canonical_direction}, but the model "
                    f"predicts OH would decrease by "
                    f"{abs(improvement_sensitivity_oh):.3f}pp per 1pp improvement"
                )
            else:
                interpretation = (
                    f"Moving {metric} in the operationally beneficial "
                    f"direction ({canonical_direction}) improves OH by "
                    f"{improvement_sensitivity_oh:.3f}pp per 1pp"
                )

            details.append({
                "metric": metric,
                "oh_change": oh_change,
                "nps_change": nps_change,
                # Backward-compatible raw experiment direction.
                "direction": direction,
                "improvement_direction": canonical_direction,
                "raw_sensitivity_oh": raw_sensitivity_oh,
                "improvement_sensitivity_oh": improvement_sensitivity_oh,
                "model_conflict": model_conflict,
                "relative_impact": rel_impact,
                "interpretation": interpretation,
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
        # Forecast AI serializes TrendAnalysis dataclasses via asdict(), so the
        # canonical key is trend_direction. The bare direction key is accepted
        # for backward compatibility with older dict-shaped inputs.
        direction = (a.get("direction") or a.get("trend_direction") or "").strip().lower()
        # Exact directional classification. The previous substring checks
        # matched "Strong Decrease" as an improvement (via "Strong") AND a
        # decline, which double-counted nps and flipped the overall direction
        # to "improving" in the presence of a strong decrease.
        if "decrease" in direction or "declin" in direction:
            declines.append(a)
        elif "increase" in direction or "improv" in direction:
            improvements.append(a)

    strongest_positive = improvements[0] if improvements else None
    strongest_negative = declines[0] if declines else None

    return {
        "direction": "stable" if not improvements and not declines else ("improving" if improvements else "declining"),
        "slope_change": [a.get("absolute_change") for a in analyses],
        "strongest_positive": {
            "metric": strongest_positive.get("metric") if strongest_positive else None,
            "change": _round(strongest_positive.get("absolute_change")) if strongest_positive else None,
            "direction": (strongest_positive.get("direction") or strongest_positive.get("trend_direction")) if strongest_positive else None,
        } if strongest_positive else None,
        "strongest_negative": {
            "metric": strongest_negative.get("metric") if strongest_negative else None,
            "change": _round(strongest_negative.get("absolute_change")) if strongest_negative else None,
            "direction": (strongest_negative.get("direction") or strongest_negative.get("trend_direction")) if strongest_negative else None,
        } if strongest_negative else None,
        "analyses": [{
            "metric": a.get("metric"),
            "direction": a.get("direction") or a.get("trend_direction"),
            "change": _round(a.get("absolute_change")),
            "pct_change": _round(a.get("percent_change")),
        } for a in analyses],
    }


def _build_agreement_detail(agreement: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build agreement detail from Forecast AI agreement output.

    Never fabricates an agreement. When no recommendation evidence is available
    (no agreement object, or an agreement without a computed score), an explicit
    ``insufficient_evidence`` state is returned rather than an apparently
    confident block of ``—``/0 values.
    """
    if not agreement:
        return {
            "available": False,
            "status": "insufficient_evidence",
            "reason": "No recommendation evidence to compute agreement/consistency.",
        }

    score = agreement.get("score")
    category_consistency = agreement.get("category_consistency")
    if score is None or category_consistency is None:
        return {
            "available": False,
            "status": "insufficient_evidence",
            "reason": "Agreement evidence is incomplete (score or consistency missing).",
        }

    conflicts = agreement.get("conflicts") or []
    return {
        "available": True,
        "status": "available",
        "score": _round(score),
        "category_consistency": _round(category_consistency),
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:5],  # Top 5
    }


def _enhance_explanation(
    explanation: dict[str, Any],
    forecast_summary: dict[str, Any],
    bayesian_detail: dict[str, Any],
    mc_detail: dict[str, Any],
    *,
    evidence_sufficient: bool = True,
    evidence_reason: str | None = None,
    preferred_name: str | None = None,
) -> dict[str, Any]:
    """Enhance the existing explanation with additional detail.

    When the decision evidence gate is insufficient the explanation is made
    explicit: the preferred scenario is a *forecast preference only* (never an
    actionable decision/recommendation), the canonical risk is ABSTAIN, and the
    recommended action is withheld.
    """
    enhanced = dict(explanation) if explanation else {}

    enhanced["bayesian"] = bayesian_detail
    enhanced["monte_carlo"] = mc_detail
    enhanced["forecast_summary"] = forecast_summary
    enhanced["decision_status"] = (
        DECISION_STATUS_AVAILABLE if evidence_sufficient else DECISION_STATUS_INSUFFICIENT
    )

    if not evidence_sufficient:
        # Preferred scenario is a forecast preference, explicitly non-actionable.
        if isinstance(enhanced.get("preferred_scenario"), dict):
            enhanced["preferred_scenario"]["actionable"] = False
        enhanced["forecast_preference"] = {
            "name": preferred_name,
            "actionable": False,
            "kind": "forecast_preference",
        }
        enhanced["why_selected"] = {
            "text": (
                f"Forecast preference only: {preferred_name or 'the top-ranked scenario'} "
                "ranked highest under the deterministic scenario-ranking policy. "
                "Canonical decision is withheld because decision evidence is insufficient."
            ),
            "actionable": False,
            "reason": evidence_reason or "Decision evidence is insufficient",
        }
        # Preserve the original explanation's main_risk fields (especially
        # diagnostic drivers) while making the canonical decision state
        # explicit.  Do not destroy an existing driver/level just because the
        # actionable decision is being withheld.
        existing_main_risk = dict(enhanced.get("main_risk") or {})
        existing_main_risk["abstain"] = True
        existing_main_risk["status"] = DECISION_STATUS_INSUFFICIENT
        existing_main_risk["reason"] = (
            evidence_reason or "Decision evidence is insufficient"
        )
        existing_main_risk["level"] = "ABSTAIN"
        existing_main_risk["canonical_level"] = "ABSTAIN"
        enhanced["main_risk"] = existing_main_risk
        enhanced["recommended_action"] = {
            "recommendation": None,
            "action": "withheld",
            "status": DECISION_STATUS_INSUFFICIENT,
            "reason": evidence_reason or "Canonical decision withheld: insufficient evidence",
        }
        # No normal decision confidence is presented on insufficient evidence.
        enhanced["decision_confidence"] = None
        # The canonical decision is ABSTAIN — never carry stale evidence that
        # claims a normal LOW/MEDIUM/HIGH risk classification on withheld data.
        enhanced["supporting_evidence"] = [
            "Canonical decision: ABSTAIN — decision withheld due to insufficient evidence."
        ]

    return enhanced


__all__ = ["build_adie_detail"]
