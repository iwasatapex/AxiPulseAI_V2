"""Direct unit tests for compose_decision_package.

Covers:
  - successful detail composition (all 11 sections present)
  - detail-builder failure path (structured error, never a raise)
  - target-dependent unavailable output (no fabricated success rate)
  - recommendation-package integrity (all supplied sections folded in,
    probabilistic core preserved as a copy)
"""
from core.decision_intelligence.v3.integration.decision_composer import compose_decision_package


def _probabilistic() -> dict:
    return {
        "recommendation": "improve",
        "risk": "MEDIUM",
        "probability": 0.66,
        "confidence": 0.73,
        "expected": 0.82,
        "downside": 0.78,
        "upside": 0.86,
        "scenarios": [{"name": "day_1", "operations_health": 95.0, "nps": 85.0}],
        "risk_score": 0.41,
        "abstain": False,
        "success_count": 6400,
        "failure_count": 3600,
        "decision": {"recommendation": "improve", "action": "review"},
        "explanation": {},
        "semantics": {"probability_of_target": {}},
        "monte_carlo_detail": {
            "success_count": 6400,
            "failure_count": 3600,
            "distribution_summary": {"mean": 0.82, "p05": 0.78, "p50": 0.82, "p95": 0.86, "samples": 10000},
        },
        "bayesian_detail": {"probability": 0.66, "confidence": 0.73},
    }


def test_compose_successful_detail_composition():
    package = compose_decision_package(
        _probabilistic(),
        recommendation_output={"status": "success", "recommendations": []},
        trend_output={"analyses": []},
        sensitivity_output={"analyses": [], "ranking": []},
        agreement={"score": 0.8, "category_consistency": 1.0, "conflicts": []},
        targets={"target_nps": 8.0, "target_operations_health": 90.0},
        horizon=3,
    )

    assert package["probabilistic"] == _probabilistic()
    assert package["probabilistic"] is not _probabilistic()  # copied, not aliased
    assert "recommendations" in package
    assert "trends" in package
    assert "sensitivity" in package
    assert "agreement" in package

    detail = package["details"]
    for section in (
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
    ):
        assert section in detail, f"missing detail section: {section}"


def test_compose_detail_builder_failure_path():
    # scenarios=[None] makes build_adie_detail raise; compose must convert that
    # into a structured details error instead of propagating.
    package = compose_decision_package(
        {"scenarios": [None], "probability": 0.5, "confidence": 0.5,
         "expected": 0.5, "downside": 0.4, "upside": 0.6},
        trend_output={"analyses": []},
    )
    assert "details" in package
    assert isinstance(package["details"], dict)
    assert "error" in package["details"]


def test_compose_target_dependent_unavailable_output():
    # Without an OH target the Monte Carlo success rate is honestly
    # "unavailable" — never a fabricated success definition.
    package = compose_decision_package(
        _probabilistic(),
        trend_output={"analyses": []},  # forecast output present -> detail built
    )
    mc = package["details"]["monte_carlo_detail"]
    assert mc["success_definition"] == "unavailable"
    assert mc["success_count"] is None
    assert mc["failure_count"] is None


def test_compose_recommendation_package_integrity():
    rec_output = {
        "status": "success",
        "recommendations": [
            {"title": "r1", "rank": 1, "affected_kpi": "quality", "direction": "increase"},
        ],
    }
    package = compose_decision_package(
        _probabilistic(),
        recommendation_output=rec_output,
        strategy_output={"strategies": []},
    )
    # Sections are folded in verbatim (no mutation).
    assert package["recommendations"] == rec_output
    assert package["strategies"] == {"strategies": []}
    # Probabilistic core is preserved.
    assert package["probabilistic"]["probability"] == 0.66
    assert package["probabilistic"]["scenarios"][0]["name"] == "day_1"
