from core.decision_intelligence.v3.integration.probabilistic_decision import (
    ProbabilisticDecisionService,
)


def test_probabilistic_decision_package():
    service = ProbabilisticDecisionService()

    result = service.analyze(
        scenarios=[
            {
                "name": "current_state",
                "probability": 0.7143,
                "confidence": 0.6806,
                "expected": 0.8192,
                "p05": 0.7525,
                "p95": 0.8846,
            },
            {
                "name": "improved_operations",
                "probability": 0.8571,
                "confidence": 0.7526,
                "expected": 0.8994,
                "p05": 0.8494,
                "p95": 0.9485,
            },
        ],
        observations=[1, 1, 0, 1, 1],
        baseline=0.82,
        uncertainty=0.04,
        samples=2000,
    )

    assert result.recommendation == "improved_operations"
    assert result.risk == "LOW"
    assert len(result.scenarios) == 2
    assert result.downside < result.upside


def test_package_serialization():
    service = ProbabilisticDecisionService()

    result = service.analyze(
        scenarios=[
            {
                "name": "current_state",
                "probability": 0.7143,
                "confidence": 0.6806,
                "expected": 0.8192,
                "p05": 0.7525,
                "p95": 0.8846,
            }
        ],
        observations=[1, 0, 1],
        baseline=0.70,
        samples=1000,
    )

    payload = service.to_dict(result)

    assert isinstance(payload, dict)
    assert payload["recommendation"] == "current_state"
    assert isinstance(payload["scenarios"], list)


def test_empty_scenarios_rejected():
    service = ProbabilisticDecisionService()

    try:
        service.analyze(
            scenarios=[],
            observations=[1, 0, 1],
            baseline=0.70,
        )
        assert False
    except ValueError:
        assert True


def _bare_scenarios():
    # No scenario-owned p05/p95/expected: the package must fall back to the
    # (bounded) decision-level Monte Carlo for the probability presentation.
    return [{"name": "day_1", "operations_health": 95.0, "nps": 85.0}]


def test_package_probability_domain_bounded():
    """Every public probability-domain field of the decision package stays
    within [0,1], and percentiles/counts remain coherent from one draw."""
    service = ProbabilisticDecisionService()
    result = service.analyze(
        scenarios=_bare_scenarios(),
        observations=[1, 1, 1, 0, 1],
        baseline=0.95,
        uncertainty=0.05,
        samples=10000,
    )

    for field in ("probability", "confidence", "expected", "downside", "upside"):
        value = getattr(result, field)
        assert 0.0 <= value <= 1.0, f"{field}={value} outside [0,1]"

    summary = result.monte_carlo_detail["distribution_summary"]
    for key in ("mean", "p05", "p50", "p95"):
        assert 0.0 <= summary[key] <= 1.0, f"{key}={summary[key]} outside [0,1]"
    assert summary["p05"] <= summary["p50"] <= summary["p95"]

    # Same-draw invariant: counts partition the same sample; bins total to it.
    assert result.monte_carlo_detail["success_count"] + result.monte_carlo_detail["failure_count"] == summary["samples"]
    assert sum(b["count"] for b in result.monte_carlo_detail["distribution"]) == summary["samples"]


def test_package_risk_level_matches_canonical():
    """The package risk level must equal the canonical classifier output for
    the same probability/confidence (one risk model)."""
    from core.decision_intelligence.v3.risk.uncertainty import UncertaintyRiskEngine

    service = ProbabilisticDecisionService()
    result = service.analyze(
        scenarios=_bare_scenarios(),
        observations=[1, 1, 1, 0, 1],
        baseline=0.95,
        uncertainty=0.05,
        samples=2000,
    )
    canonical = UncertaintyRiskEngine.classify_level(result.probability, result.confidence)
    assert result.risk == canonical


def test_package_deterministic_repeat():
    """Same inputs + same seed reproduce an identical decision package."""
    service = ProbabilisticDecisionService()

    def run():
        return service.analyze(
            scenarios=_bare_scenarios(),
            observations=[1, 1, 1, 0, 1],
            baseline=0.95,
            uncertainty=0.05,
            samples=2000,
        )

    a, b = run(), run()
    assert a.probability == b.probability
    assert a.risk == b.risk
    assert a.downside == b.downside
    assert a.upside == b.upside
    assert a.monte_carlo_detail["distribution_summary"] == b.monte_carlo_detail["distribution_summary"]
