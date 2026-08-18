"""
Focused tests for the canonical ADIE V3 decision-detail builder
(``core.decision_intelligence.v3.synthesis.decision_detail``).

Validates the Phase 16 richer decision-intelligence output:
  - top-3 recommendation ranking (never fabricates to reach 3)
  - forecast OH/NPS ranges + per-day table
  - Bayesian detail (decision probability, posterior ranges, NPS 0-10)
  - Monte Carlo detail (success/failure counts and %, distribution bins)
  - risk / sensitivity / trend / agreement / scenario-comparison detail
  - enhanced explanation
  - None / NaN / INF input safety (no regression to NaN/INF)
  - backward compatibility (existing fields preserved)
"""
from __future__ import annotations

import math
import importlib
from dataclasses import asdict

import pytest

from core.decision_intelligence.v3.synthesis.decision_detail import build_adie_detail
from core.forecast_ai.sensitivity.models import SensitivityAnalysis
from core.forecast_ai.trends.models import TrendAnalysis


# =====================================================================
# Fixtures
# =====================================================================

def _nps_distribution(center: int = 8) -> dict:
    """Realistic 0..10 posterior over business-NPS scores."""
    weights = {score: math.exp(-0.5 * (score - center) ** 2) for score in range(0, 11)}
    total = sum(weights.values())
    return {score: weight / total for score, weight in weights.items()}


def _scenarios(n: int = 5) -> list[dict]:
    scenarios = []
    for i in range(n):
        oh = 95.0 - i * 1.5
        nps = 82.0 - i * 2.0
        scenarios.append({
            "name": f"forecast_day_{i}",
            "_predicted": True,
            "operations_health": oh,
            "nps": nps,
            "probability": max(0.6 - i * 0.05, 0.1),
            "confidence": max(0.8 - i * 0.06, 0.1),
            "delta_oh": 1.5 if i == 0 else 0.0,
            "risk_severity": 0.10 + i * 0.02,
            "expected": oh,
            "p05": oh - 4.0,
            "p50": oh,
            "p95": oh + 4.0,
            # Unambiguous semantics:
            #   expected_score : 0..10 mean survey score
            #   score_p05/p95  : 0..10 score quantiles
            #   nps_p05/p95    : -100..100 NPS quantiles (Monte Carlo)
            "expected_score": 8.0 - i * 0.1,
            "score_p05": 6.0,
            "score_p95": 10.0,
            "nps_p05": nps - 6.0,
            "nps_p95": nps + 6.0,
            "bayesian_score_distribution": _nps_distribution(center=8 - i),
        })
    return scenarios


def _mc_detail(success_count: int = 6400, failure_count: int = 3600) -> dict:
    # Distribution is on the normalized [0,1] health-score scale (matches the
    # production decision-level Monte Carlo). Samples total 10,000.
    total = success_count + failure_count
    return {
        "success_count": success_count,
        "failure_count": failure_count,
        "distribution_summary": {
            "mean": 0.82,
            "p05": 0.78,
            "p50": 0.82,
            "p95": 0.86,
            "probability_positive": success_count / total if total else 0.0,
            "samples": total,
            "uncertainty": 0.03,
        },
        "distribution": [
            {"bin_start": 0.76, "bin_end": 0.78, "count": 1200, "probability": 0.12},
            {"bin_start": 0.78, "bin_end": 0.80, "count": 1800, "probability": 0.18},
            {"bin_start": 0.80, "bin_end": 0.82, "count": 2400, "probability": 0.24},
            {"bin_start": 0.82, "bin_end": 0.84, "count": 2200, "probability": 0.22},
            {"bin_start": 0.84, "bin_end": 0.86, "count": 2400, "probability": 0.24},
        ],
    }


def _bayesian_detail() -> dict:
    return {
        "probability": 0.66,
        "confidence": 0.73,
        "posterior_mean": 0.66,
        "posterior_std": 0.04,
        "credible_interval_lower": 0.58,
        "credible_interval_upper": 0.74,
        "credible_level": 0.95,
        "samples": 10_000,
    }


def _policy_decision() -> dict:
    return {
        "recommendation": "improve",
        "rationale": "forecast below target",
        "target": {"target_nps": 8.0, "target_operations_health": 90.0},
    }


def _package(**overrides) -> dict:
    package = {
        "recommendation": "improve",
        "risk": "MEDIUM",
        "probability": 0.66,
        "confidence": 0.73,
        "expected": 82.0,
        "downside": 78.0,
        "upside": 86.0,
        "scenarios": _scenarios(),
        "risk_score": 0.41,
        "abstain": False,
        "success_count": 6400,
        "failure_count": 3600,
        "decision": _policy_decision(),
        "explanation": {
            "current_state": {"operations_health": 96.0, "nps": 84.0},
            "preferred_scenario": {"name": "forecast_day_0", "rank": 1, "score": 0.9},
            "why_preferred": {"policy": "best scenario maximizes expected OH"},
            "main_risk": {"driver": "attendance decline"},
            "uncertainty": {"p05": 78.0, "p95": 86.0},
            "decision_changers": {"prob_threshold": 0.6},
        },
        "semantics": {
            "probability_interpretation": "posterior mean of normalized health observations",
            "confidence_interpretation": "1 - normalized posterior std",
            "monte_carlo_samples": 10_000,
            "probability_of_target": {
                "nps": {"target": 8.0, "expected_nps": 8.1, "probability_promoter_score": 0.62},
                "operations_health": {"target": 90.0, "probability": 0.58},
            },
        },
        "monte_carlo_detail": _mc_detail(),
        "bayesian_detail": _bayesian_detail(),
    }
    package.update(overrides)
    return package


def _recommendation_output(recs: list[dict]) -> dict:
    return {"status": "success", "recommendations": recs}


def _rec(title: str, *, score: float = 1.0, priority: str = "high") -> dict:
    return {
        "id": f"rec_{title}",
        "title": title,
        "description": "advisory",
        "category": "operations",
        "priority": priority,
        "difficulty": "medium",
        "estimated_oh_gain": 1.2,
        "estimated_nps_gain": 0.5,
        "confidence": 0.75,
        "actions": ["increase quality training"],
        "reasoning": f"evidence for {title}",
        "optimization_score": score,
        "metadata": {},
        "target_kpi": "quality",
        "direction": "increase",
    }

# =====================================================================
# Backward compatibility: existing fields preserved
# =====================================================================

def test_backward_compatibility_fields_preserved():
    pkg = _package()
    detail = build_adie_detail(pkg)
    # New canonical sections are all present.
    for section in [
        "recommendations",
        "forecast_summary",
        "scenario_comparison",
        "bayesian_detail",
        "monte_carlo_detail",
        "risk_detail",
        "sensitivity_detail",
        "trend_detail",
        "agreement",
        "explanation",
        "best_scenario",
    ]:
        assert section in detail, f"missing detail section: {section}"
    # Existing package primitives untouched (dict in -> fresh dict out; no mutation).
    assert pkg["scenarios"][0]["operations_health"] == 95.0
    assert pkg["monte_carlo_detail"]["success_count"] == 6400
    # Not a dict copy of the package — recommendations are a standalone list.
    assert isinstance(detail["recommendations"], list)


def test_build_adie_detail_does_not_mutate_inputs():
    pkg = _package()
    scenarios_before = [dict(s) for s in pkg["scenarios"]]
    detail = build_adie_detail(
        pkg,
        recommendation_output=_recommendation_output([_rec("a")]),
        trend_output={"analyses": []},
        sensitivity_output={"analyses": [], "ranking": []},
        agreement={"score": 0.8, "category_consistency": 1.0, "conflicts": []},
        targets={"target_nps": 8.0, "target_operations_health": 90.0},
        observed=96.0,
        observed_metrics=["operations_health", "nps"],
        horizon=5,
    )
    assert pkg["scenarios"] == scenarios_before
    assert "details" not in pkg


# =====================================================================
# Top-3 recommendation ranking
# =====================================================================

def test_recommendations_up_to_three_ranked():
    recs = [
        _rec("most", score=0.95),
        _rec("middle", score=0.70),
        _rec("least", score=0.45),
        _rec("fourth", score=0.20),
    ]
    detail = build_adie_detail(
        _package(), recommendation_output=_recommendation_output(recs)
    )
    top = detail["recommendations"]
    assert len(top) == 3  # >3 genuine recs -> capped at top 3
    assert [r["rank"] for r in top] == [1, 2, 3]
    assert top[0]["action"] == "most"
    assert top[1]["action"] == "middle"
    assert top[2]["action"] == "least"
    assert top[0]["affected_kpi"] == "quality"
    assert top[0]["direction"] == "increase"
    assert top[0]["confidence"] == 0.75
    assert top[0]["risk"] in {"LOW", "MEDIUM", "HIGH"}
    # expected effect is exposed only when available
    assert top[0]["expected_effect"]["oh_gain"] == 1.2


def test_recommendations_single():
    detail = build_adie_detail(
        _package(), recommendation_output=_recommendation_output([_rec("only")])
    )
    assert len(detail["recommendations"]) == 1
    assert detail["recommendations"][0]["rank"] == 1
    assert detail["recommendations"][0]["evidence"]


def test_recommendations_empty_no_fabrication():
    # No recommendation outputs AND no targets -> zero recommendations.
    detail = build_adie_detail(_package())
    assert detail["recommendations"] == []


def test_recommendations_missing_target_no_fabrication():
    # recommendation engine skipped -> scenario-fallback recs must NOT be
    # invented without targets.
    detail = build_adie_detail(_package(), recommendation_output=None, targets=None)
    assert detail["recommendations"] == []


def test_recommendations_skipped_optimizer():
    detail = build_adie_detail(
        _package(),
        recommendation_output={"status": "skipped", "reason": "missing_target", "recommendations": []},
        targets={"target_nps": 8.0},
    )
    # skipped optimizer -> no fabricated recommendations despite a target
    assert detail["recommendations"] == []


def test_recommendations_conflicting_preserves_evidence():
    conflicting = [
        _rec("increase_attendance", score=0.8, priority="high"),
        _rec("decrease_transfer", score=0.7, priority="high"),
        _rec("increase_quality", score=0.6, priority="medium"),
    ]
    detail = build_adie_detail(
        _package(), recommendation_output=_recommendation_output(conflicting)
    )
    assert len(detail["recommendations"]) == 3
    # Conflicting recommendations are preserved as-is (with evidence) — the
    # builder never merges or discards genuine recs to hide conflict.
    actions = {r["action"] for r in detail["recommendations"]}
    assert actions == {"increase_attendance", "decrease_transfer", "increase_quality"}
    assert all(r["evidence"] for r in detail["recommendations"])


def test_recommendations_rank_present_and_unique():
    recs = [_rec(f"r{i}", score=(0.9 - i * 0.1)) for i in range(6)]
    detail = build_adie_detail(
        _package(), recommendation_output=_recommendation_output(recs)
    )
    top = detail["recommendations"]
    assert len(top) == 3
    assert [r["rank"] for r in top] == [1, 2, 3]

# =====================================================================
# Surface / module contract
# =====================================================================

def test_module_contract():
    module = importlib.import_module(
        "core.decision_intelligence.v3.synthesis.decision_detail"
    )
    assert hasattr(module, "build_adie_detail")
    assert module.__all__ == ["build_adie_detail"]
# =====================================================================
# Forecast summary / ranges
# =====================================================================

def test_forecast_summary_oh_nps_ranges():
    detail = build_adie_detail(
        _package(), observed=96.0,
        observed_metrics=["operations_health", "nps"], horizon=5,
    )
    fs = detail["forecast_summary"]
    assert fs["oh_range"]["min"] == pytest.approx(89.0, abs=1e-6)   # 95.0 - 4*1.5
    assert fs["oh_range"]["max"] == pytest.approx(95.0, abs=1e-6)
    assert fs["oh_range"]["expected"] == pytest.approx(95.0, abs=1e-6)
    assert fs["nps_range"]["min"] == pytest.approx(74.0, abs=1e-6)  # 82.0 - 4*2.0
    assert fs["nps_range"]["max"] == pytest.approx(82.0, abs=1e-6)
    assert fs["nps_range"]["expected"] == pytest.approx(82.0, abs=1e-6)
    assert fs["best_day"]["day_index"] == 1
    assert fs["worst_day"]["day_index"] == 5
    assert fs["expected_day"]["day_index"] == 1
    assert fs["scenario_count"] == 5
    assert fs["horizon_days"] == 5


def test_forecast_per_day_table_columns():
    detail = build_adie_detail(_package(), horizon=5)
    table = detail["forecast_summary"]["per_day_table"]
    assert len(table) == 5
    first = table[0]
    for col in ["day", "oh", "nps", "confidence", "risk", "_predicted",
                "score_p05", "score_p95", "expected_score",
                "nps_p05", "nps_p95", "nps_distribution"]:
        assert col in first, f"missing per-day column: {col}"
    assert first["day"] == 1
    assert first["_predicted"] is not None
    # expected_score is a 0..10 score, never the -100..100 business NPS.
    assert 0.0 <= first["expected_score"] <= 10.0
    # score_p05/p95 are 0..10 quantiles.
    assert 0.0 <= first["score_p05"] <= 10.0
    assert 0.0 <= first["score_p95"] <= 10.0


def test_forecast_summary_empty_scenarios():
    detail = build_adie_detail(_package(scenarios=[]), recommendation_output=None)
    assert "note" in detail["forecast_summary"]
    assert detail["forecast_summary"]["note"]


# =====================================================================
# Bayesian detail
# =====================================================================

def test_bayesian_detail_exposes_existing_fields():
    detail = build_adie_detail(_package())
    bay = detail["bayesian_detail"]
    assert bay["decision_probability"] == 0.66
    assert bay["confidence"] == 0.73
    assert bay["posterior_ranges"]["mean"] == 0.66
    assert bay["posterior_ranges"]["std"] == 0.04
    cred = bay["posterior_ranges"]["credible_interval"]
    assert cred["lower"] == 0.58 and cred["upper"] == 0.74 and cred["level"] == 0.95
    assert bay["nps_0_10_distribution"]
    assert bay["nps_expected_business"] is not None
    assert bay["interpretation"]


def test_bayesian_detail_missing_fields_no_crash():
    pkg = _package(bayesian_detail={}, semantics={})
    detail = build_adie_detail(pkg)
    bay = detail["bayesian_detail"]
    assert bay["decision_probability"] is None
    assert bay["posterior_ranges"]["mean"] is None
    assert "interpretation" in bay


def test_bayesian_target_probabilities_preserved():
    detail = build_adie_detail(_package())
    bay = detail["bayesian_detail"]
    # Real target probabilities live in semantics and remain intact; they
    # are never re-computed or replaced by a generic "probability of success".
    assert detail["explanation"]["bayesian"]["decision_probability"] == 0.66


# =====================================================================
# Monte Carlo detail
# =====================================================================

def test_mc_success_count_and_percentage_target_based():
    pkg = _package()
    pkg["monte_carlo_detail"] = _mc_detail(success_count=6400, failure_count=3600)
    # target_oh on 0-100 scale; success = simulated OH outcome satisfies target
    # (P(OH >= 80/100) derived from the single existing MC distribution).
    mc = build_adie_detail(pkg, targets={"target_operations_health": 80.0})["monte_carlo_detail"]
    assert mc["total_samples"] == 10_000
    assert mc["success_count"] + mc["failure_count"] == 10_000
    assert 0.0 <= mc["success_percentage"] <= 100.0
    assert abs((mc["success_percentage"] + mc["failure_percentage"]) - 100.0) < 1e-6
    assert "target_oh" in mc["success_definition"]
    assert "P(OH >=" in mc["interpretation"]
    assert mc["p05"] == pytest.approx(0.78, abs=1e-6)
    assert mc["p50"] == pytest.approx(0.82, abs=1e-6)
    assert mc["p95"] == pytest.approx(0.86, abs=1e-6)
    assert mc["expected_value"] == pytest.approx(0.82, abs=1e-6)
    assert mc["distribution_bins"]
    assert mc["interpretation"]


def test_mc_success_unavailable_without_target():
    # Without a target, success rate is explicitly UNAVAILABLE (not "sample>0").
    pkg = _package()
    pkg["monte_carlo_detail"] = _mc_detail(success_count=6400, failure_count=3600)
    mc = build_adie_detail(pkg)["monte_carlo_detail"]
    assert mc["success_count"] is None
    assert mc["failure_count"] is None
    assert mc["success_percentage"] is None
    assert mc["failure_percentage"] is None
    assert mc["success_definition"] == "unavailable"
    assert "unavailable" in mc["interpretation"]


def test_mc_zero_success_target_based():
    # target_oh=95 (ratio 0.95) far above the distribution -> ~0% success.
    pkg = _package()
    pkg["monte_carlo_detail"] = _mc_detail(success_count=6400, failure_count=3600)
    mc = build_adie_detail(pkg, targets={"target_operations_health": 95.0})["monte_carlo_detail"]
    assert mc["success_count"] == 0
    assert mc["failure_count"] == 10_000
    assert mc["success_percentage"] == pytest.approx(0.0, abs=1e-6)
    assert mc["failure_percentage"] == pytest.approx(100.0, abs=1e-6)


def test_mc_hundred_percent_success_target_based():
    # target_oh=70 (ratio 0.70) far below the distribution -> ~100% success.
    pkg = _package()
    pkg["monte_carlo_detail"] = _mc_detail(success_count=6400, failure_count=3600)
    mc = build_adie_detail(pkg, targets={"target_operations_health": 70.0})["monte_carlo_detail"]
    assert mc["success_count"] == 10_000
    assert mc["failure_count"] == 0
    assert mc["success_percentage"] == pytest.approx(100.0, abs=1e-6)
    assert mc["failure_percentage"] == pytest.approx(0.0, abs=1e-6)


def test_mc_missing_samples_no_crash():
    pkg = _package(monte_carlo_detail={})
    mc = build_adie_detail(pkg, targets={"target_operations_health": 80.0})["monte_carlo_detail"]
    assert mc["total_samples"] is None
    assert mc["success_count"] is None
    assert mc["success_percentage"] is None
# =====================================================================
# Risk detail
# =====================================================================

def test_risk_detail_exposes_fields():
    detail = build_adie_detail(_package())
    risk = detail["risk_detail"]
    assert risk["level"] == "MEDIUM"
    assert risk["score"] == pytest.approx(0.41, abs=1e-6)
    assert risk["confidence"] == pytest.approx(0.73, abs=1e-6)
    assert risk["downside"] == pytest.approx(78.0, abs=1e-6)
    assert risk["upside"] == pytest.approx(86.0, abs=1e-6)
    assert risk["abstain"] is False
    assert risk["drivers"]
    assert risk["threshold_policy"]


# =====================================================================
# Sensitivity detail (5 KPIs)
# =====================================================================

def _sensitivity_output():
    metrics = {
        "quality": 1.4,
        "competency": 1.1,
        "attendance": 0.9,
        "release": 0.4,
        "transfer": -0.6,
    }
    analyses = []
    for i, (metric, sens) in enumerate(metrics.items()):
        analyses.append({
            "metric": metric,
            "oh_change": sens,
            "nps_change": sens * 0.5,
            "sensitivity_oh": sens,
            "sensitivity_nps": sens * 0.4,
            "elasticity_oh": 1.0,
            "elasticity_nps": 1.0,
            "classification": "High" if abs(sens) > 1.0 else "Medium",
            "rank": i,
        })
    return {"analyses": analyses, "ranking": analyses}


def test_sensitivity_detail_all_five_kpis():
    detail = build_adie_detail(_package(), sensitivity_output=_sensitivity_output())
    sens = detail["sensitivity_detail"]
    metrics = {m["metric"] for m in sens["metrics"]}
    assert metrics == {"quality", "competency", "attendance", "release", "transfer"}
    by_metric = {m["metric"]: m for m in sens["metrics"]}
    assert by_metric["quality"]["oh_change"] == 1.4
    assert by_metric["quality"]["relative_impact"] == "high"
    assert by_metric["transfer"]["direction"] == "decrease"
    assert sens["ranking"]


def test_sensitivity_detail_missing():
    detail = build_adie_detail(_package())
    sens = detail["sensitivity_detail"]
    assert "note" in sens


# =====================================================================
# Trend detail
# =====================================================================

def _trend_output():
    return {
        "analyses": [
            {"metric": "operations_health", "direction": "Decrease",
             "absolute_change": -2.0, "percent_change": -2.1},
            {"metric": "quality", "direction": "Decrease",
             "absolute_change": -0.8, "percent_change": -0.9},
            {"metric": "nps", "direction": "Decrease",
             "absolute_change": -3.0, "percent_change": -3.4},
        ],
    }


def test_trend_detail_exposes_direction_and_extremes():
    detail = build_adie_detail(_package(), trend_output=_trend_output())
    trend = detail["trend_detail"]
    assert trend["direction"] == "declining"
    assert trend["strongest_positive"] is None
    assert trend["strongest_negative"]["metric"] in {"operations_health", "nps"}
    assert trend["analyses"]


def test_trend_detail_missing():
    detail = build_adie_detail(_package(), trend_output=None)
    trend = detail["trend_detail"]
    assert "note" in trend


def test_trend_detail_stable_when_no_clear_direction():
    detail = build_adie_detail(
        _package(),
        trend_output={"analyses": [{"metric": "nps", "direction": "Flat",
                                    "absolute_change": 0.0}]},
    )
    assert detail["trend_detail"]["direction"] == "stable"


# =====================================================================
# Production-shaped (asdict) sensitivity / trend input
# =====================================================================

def test_sensitivity_detail_uses_production_asdict_shape():
    """Forecast AI serializes analyses via ``dataclasses.asdict``, whose keys
    are ``sensitivity_score_oh`` / ``sensitivity_score_nps`` (not
    ``sensitivity_oh`` / ``sensitivity_nps``). The ADIE builder must read the
    canonical keys and never fall back to "Unknown impact" for a metric that
    actually has a sensitivity score."""
    sens = build_adie_detail(
        _package(),
        sensitivity_output={
            "analyses": [
                asdict(SensitivityAnalysis(
                    metric="quality",
                    baseline_output_oh=94.75, baseline_output_nps=85.48,
                    modified_output_oh=94.66, modified_output_nps=85.89,
                    operations_health_change=-0.0913, nps_change=0.405,
                    sensitivity_score_oh=-0.0913, sensitivity_score_nps=0.405,
                    elasticity_oh=0.0, elasticity_nps=0.0,
                )),
            ],
            "ranking": [],
        },
    )["sensitivity_detail"]
    m = sens["metrics"][0]
    assert m["metric"] == "quality"
    assert m["direction"] == "decrease"            # oh_change < 0
    assert m["relative_impact"] == "low"           # |sens| 0.09 < 0.2
    assert "Unknown impact" not in m["interpretation"]
    assert "0.091" in m["interpretation"]


def test_sensitivity_detail_relative_impact_magnitude_based():
    sens = build_adie_detail(
        _package(),
        sensitivity_output={
            "analyses": [
                asdict(SensitivityAnalysis(
                    metric="quality",
                    baseline_output_oh=90.0, baseline_output_nps=80.0,
                    modified_output_oh=91.4, modified_output_nps=80.0,
                    operations_health_change=1.4, nps_change=0.0,
                    sensitivity_score_oh=1.4, sensitivity_score_nps=0.0,
                    elasticity_oh=0.0, elasticity_nps=0.0,
                )),
            ],
            "ranking": [],
        },
    )["sensitivity_detail"]
    m = sens["metrics"][0]
    assert m["relative_impact"] == "high"          # |1.4| > 0.5
    assert m["direction"] == "increase"


# =====================================================================
# Risk-consistency invariant (one canonical risk model)
# =====================================================================

def test_recommendation_risk_label_matches_canonical():
    """Top-3 recommendation risk labels must equal the canonical
    UncertaintyRiskEngine level for the same probability/confidence pair."""
    from core.decision_intelligence.v3.risk.uncertainty import UncertaintyRiskEngine

    cases = [
        (0.45, 0.90),   # previously HIGH in _assess_risk, canonical MEDIUM
        (0.55, 0.95),
        (0.80, 0.80),
        (0.30, 0.30),
        (0.65, 0.65),
        (0.34, 0.90),
        (0.35, 0.90),
        (0.59, 0.59),
        (0.60, 0.60),
    ]
    for prob, conf in cases:
        rec = _rec("r", score=0.9)
        rec["probability"] = prob
        rec["confidence"] = conf
        detail = build_adie_detail(
            _package(),
            recommendation_output=_recommendation_output([rec]),
        )
        label = detail["recommendations"][0]["risk"]
        canonical = UncertaintyRiskEngine.classify_level(prob, conf)
        assert label == canonical, (
            f"prob={prob} conf={conf}: recommendation={label} canonical={canonical}"
        )


def test_assess_risk_missing_inputs_neutral_medium():
    """Non-finite/missing inputs map to the neutral MEDIUM label, never to
    alternate thresholds."""
    from core.decision_intelligence.v3.synthesis.decision_detail import _assess_risk

    assert _assess_risk(None, None) == "MEDIUM"
    assert _assess_risk(float("nan"), 0.9) == "MEDIUM"
    assert _assess_risk(0.8, float("inf")) == "MEDIUM"
    assert _assess_risk(0.8, 0.9) == "LOW"


def test_trend_detail_uses_production_asdict_shape():
    """``trend_direction`` is the canonical asdict key; reading only
    ``direction`` must not leave the per-metric table blank or hide a real
    decline in the overall direction / strongest-movement fields."""
    trend = build_adie_detail(
        _package(),
        trend_output={
            "analyses": [
                asdict(TrendAnalysis(
                    metric="nps", trend_direction="Decrease", trend_strength="Weak",
                    moving_average=[85.4], minimum=83.33, maximum=85.48, mean=84.05,
                    median=83.33, variance=1.0, standard_deviation=1.0, volatility="Low",
                    absolute_change=-2.15, percent_change=-2.5152, pattern="Decreasing",
                )),
            ],
        },
    )["trend_detail"]
    assert trend["direction"] == "declining"
    assert trend["strongest_negative"]["metric"] == "nps"
    assert trend["strongest_negative"]["direction"] == "Decrease"
    assert trend["analyses"][0]["direction"] == "Decrease"


# =====================================================================
# Agreement / conflicts
# =====================================================================

def test_agreement_detail_conflicts():
    agreement = {
        "score": 0.62,
        "category_consistency": 0.8,
        "conflicts": [
            {"a": "increase_attendance", "b": "decrease_attendance",
             "why": "contradictory"},
            {"a": "increase_quality", "b": "reduce_costs", "why": "tension"},
        ],
    }
    detail = build_adie_detail(_package(), agreement=agreement)
    ag = detail["agreement"]
    assert ag["score"] == pytest.approx(0.62, abs=1e-6)
    assert ag["category_consistency"] == 0.8
    assert ag["conflict_count"] == 2
    assert len(ag["conflicts"]) == 2


def test_agreement_detail_missing():
    detail = build_adie_detail(_package(), agreement=None)
    ag = detail["agreement"]
    # Explicit insufficient-evidence state, never an apparently confident block.
    assert ag["available"] is False
    assert ag["status"] == "insufficient_evidence"


# =====================================================================
# Scenario comparison ranking
# =====================================================================

def test_scenario_comparison_ranked_with_detail():
    detail = build_adie_detail(_package())
    sc = detail["scenario_comparison"]
    assert len(sc) == 5
    assert sc[0]["name"] == "forecast_day_0"
    assert sc[0]["oh"] == pytest.approx(95.0, abs=1e-6)
    assert sc[0]["probability"] is not None
    assert sc[0]["p05"] is not None
    assert sc[0]["p50"] is not None
    assert sc[0]["p95"] is not None
    assert sc[0]["_predicted"] is True
    assert detail["best_scenario"]["name"] == "forecast_day_0"


# =====================================================================
# Explanation
# =====================================================================

def test_explanation_enhanced_without_overwrite():
    detail = build_adie_detail(_package())
    exp = detail["explanation"]
    # Existing explanation content preserved.
    assert exp["current_state"]["operations_health"] == 96.0
    assert exp["main_risk"]["driver"] == "attendance decline"
    # Enhanced with derived surfaces.
    assert exp["forecast_summary"]["oh_range"]["expected"] == pytest.approx(95.0, abs=1e-6)
    assert exp["bayesian"]["decision_probability"] == 0.66
    assert exp["monte_carlo"]["total_samples"] == 10_000
    # success rate is explicitly unavailable without a target (never invented)
    assert exp["monte_carlo"]["success_percentage"] is None


def test_explanation_missing_ok():
    detail = build_adie_detail(_package(explanation={}))
    assert detail["explanation"]["forecast_summary"]["scenario_count"] == 5


# =====================================================================
# None / NaN / INF input safety
# =====================================================================

def test_nan_inf_inputs_sanitized():
    scenarios = [
        {
            "name": "bad_day",
            "operations_health": math.nan,
            "nps": math.inf,
            "confidence": None,
            "probability": math.nan,
        },
        {
            "name": "good_day",
            "operations_health": 90.0,
            "nps": 80.0,
            "confidence": 0.7,
            "probability": 0.6,
        },
    ]
    package = _package(
        scenarios=scenarios,
        risk_score=math.nan,
        probability=math.inf,
        confidence=None,
        monte_carlo_detail={
            "success_count": math.nan,
            "failure_count": None,
            "distribution_summary": {
                "mean": math.inf, "p05": None, "p50": None,
                "p95": None, "samples": None,
            },
        },
    )
    detail = build_adie_detail(package)

    def assert_no_nonfinite(obj: dict, path: str = "") -> None:
        for key, value in obj.items():
            if isinstance(value, dict):
                assert_no_nonfinite(value, f"{path}.{key}")
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict):
                        assert_no_nonfinite(item, f"{path}.{key}[]")
                    elif isinstance(item, float) and not math.isfinite(item):
                        raise AssertionError(
                            f"non-finite in {path}.{key}: {item}"
                        )
            elif isinstance(value, float):
                assert math.isfinite(value), f"non-finite in {path}.{key}: {value}"

    fs = detail["forecast_summary"]
    assert fs["oh_range"]["min"] == pytest.approx(90.0, abs=1e-6)
    assert fs["oh_range"]["max"] == pytest.approx(90.0, abs=1e-6)
    assert_no_nonfinite(detail["forecast_summary"])
    assert_no_nonfinite(detail["bayesian_detail"])
    assert_no_nonfinite(detail["monte_carlo_detail"])
    assert_no_nonfinite(detail["risk_detail"])


def test_none_inputs_do_not_raise():
    detail = build_adie_detail(
        None or {},
        recommendation_output=None,
        strategy_output=None,
        trend_output=None,
        sensitivity_output=None,
        agreement=None,
        targets=None,
        observed=None,
        observed_metrics=None,
        horizon=None,
    )
    assert detail["recommendations"] == []
    assert "note" in detail["forecast_summary"]
    assert "note" in detail["sensitivity_detail"]
    assert "note" in detail["trend_detail"]
    # Agreement with no evidence returns an explicit insufficient-evidence state.
    assert detail["agreement"]["available"] is False
    assert detail["agreement"]["status"] == "insufficient_evidence"
    assert detail["monte_carlo_detail"]["success_percentage"] is None
# =====================================================================
# Phase-16 semantics: _predicted bool, range labels, ranking evidence
# =====================================================================

def test_forecast_predicted_boolean_not_derived():
    """_predicted in the per-day table is the real source boolean (True/False)."""
    pkg = _package()
    # scenarios[0] is predicted; force an observed day to also appear.
    pkg["scenarios"] = [
        {**s, "_predicted": True} for s in _scenarios(4)
    ] + [{**_scenarios(1)[0], "_predicted": False, "name": "observed_day"}]
    detail = build_adie_detail(pkg, horizon=5)
    table = detail["forecast_summary"]["per_day_table"]
    # The observed day must show _predicted=False (real value, never NULL).
    observed_rows = [r for r in table if r["_predicted"] is False]
    predicted_rows = [r for r in table if r["_predicted"] is True]
    assert observed_rows, "an observed (_predicted=False) day should be present"
    assert predicted_rows
    for r in table:
        assert r["_predicted"] in (True, False), "never NULL when source has it"


def test_forecast_range_labels_distinguish_point_vs_probabilistic():
    """OH/NPS range is labelled a POINT forecast range (not a confidence interval)."""
    detail = build_adie_detail(_package(), horizon=5)
    fs = detail["forecast_summary"]
    assert fs["oh_range"]["type"] == "point_forecast_range"
    assert "point" in fs["oh_range"]["label"].lower()
    assert fs["nps_range"]["type"] == "point_forecast_range"
    # Separate probabilistic interval object is exposed.
    assert "probabilistic_interval" in fs
    assert fs["probabilistic_interval"]["oh_p05"] is not None
    assert fs["probabilistic_interval"]["oh_p95"] is not None


def test_forecast_best_worst_day_use_actual_rank():
    """best/worst/expected day reflect the actual ranked scenario (rank + score)."""
    ranked = [
        {**s, "rank": i + 1, "score": 0.9 - i * 0.1,
         "evidence": {"available": ["performance", "confidence"]}}
        for i, s in enumerate(_scenarios(4))
    ]
    detail = build_adie_detail(_package(scenarios=ranked), horizon=4)
    fs = detail["forecast_summary"]
    assert fs["best_day"]["rank"] == 1
    assert fs["best_day"]["score"] == pytest.approx(0.9, abs=1e-6)
    assert fs["best_day"]["name"] == "forecast_day_0"
    assert fs["worst_day"]["rank"] == 4
    assert fs["best_day"]["factors"] == ["performance", "confidence"]


def test_scenario_comparison_exposes_rank_score_and_factors():
    ranked = [
        {**s, "rank": i + 1, "score": 0.9 - i * 0.1,
         "evidence": {"available": ["performance"], "policy": "w*p"}}
        for i, s in enumerate(_scenarios(3))
    ]
    detail = build_adie_detail(_package(scenarios=ranked))
    sc = detail["scenario_comparison"]
    assert sc[0]["rank"] == 1
    assert sc[0]["score"] == pytest.approx(0.9, abs=1e-6)
    assert sc[0]["ranking_factors"] == ["performance"]
    assert sc[0]["ranking_policy"] == "w*p"


def test_mc_success_definition_field_present():
    """success_definition + interpretation always present; never sample>0 claim."""
    detail = build_adie_detail(_package(), targets={"target_operations_health": 80.0})
    mc = detail["monte_carlo_detail"]
    assert mc["success_definition"].startswith("simulated OH outcome >=")
    assert "sample > 0" not in mc["interpretation"].lower()
    # without target -> unavailable
    mc2 = build_adie_detail(_package())["monte_carlo_detail"]
    assert mc2["success_definition"] == "unavailable"
    assert "unavailable" in mc2["interpretation"].lower()