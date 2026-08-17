"""Bayesian semantics and NPS distribution consumption (Phases 7, 9)."""

from core.decision_intelligence.v3.bayesian import inference as bayes_utils
from core.decision_intelligence.v3.integration.probabilistic_decision import (
    _aggregate_target_probability,
)
from core.decision_intelligence.v3.integration.probabilistic_decision import (
    ProbabilisticDecisionService,
)


def test_score_distribution_probability_at_or_above():
    dist = {"score_8": 0.2, "score_9": 0.5, "score_10": 0.3}
    assert bayes_utils.score_distribution_probability_at_or_above(dist, 9) == 0.8
    assert bayes_utils.score_distribution_probability_at_or_above(dist, 10) == 0.3
    assert bayes_utils.score_distribution_probability_at_or_above({}, 9) == 0.0


def test_expected_business_nps_and_promoter():
    dist = {"score_8": 0.2, "score_9": 0.5, "score_10": 0.3}
    # promoters (9,10) = 0.8, detractors (0..6) = 0 -> NPS = 80
    assert bayes_utils.expected_nps_business(dist) == 80.0
    assert bayes_utils.promoter_probability(dist) == 0.8


def test_target_probability_is_honest_not_fabricated():
    class MC:
        mean = 0.82
        p05 = 0.70
        p95 = 0.94

    dist = {"score_8": 0.2, "score_9": 0.5, "score_10": 0.3}
    result = _aggregate_target_probability(
        MC(),
        {"target_oh": 84.0, "target_nps": 75.0},
        {"operations_health": 84.0, "nps": 80.0, "bayesian_score_distribution": dist},
    )
    assert result["nps"]["expected_nps"] == 80.0
    assert result["nps"]["probability_promoter_score"] == 0.8
    # P(business-NPS >= target) is explicitly unavailable, not a fake 0.0.
    assert result["nps"]["probability_of_target_nps"] is None
    assert result["operations_health"]["probability"] > 0.0


def test_bayesian_interpretation_present_and_documented():
    service = ProbabilisticDecisionService()
    result = service.analyze(
        scenarios=[{"name": "current_state", "probability": 0.7, "confidence": 0.6}],
        observations=[1, 0, 1],
        baseline=0.7,
        samples=1000,
    )
    semantics = result.semantics
    assert "probability_interpretation" in semantics
    assert "health-score posterior" in semantics["probability_interpretation"]
    assert "confidence_interpretation" in semantics
    assert "probability_of_target" in semantics


def test_confidence_is_sample_driven_and_documented():
    # Two identical sample counts give identical confidence even with
    # opposite observations (documented interpretation: confidence reflects
    # evidence mass, not extremity).
    service = ProbabilisticDecisionService()
    a = service.analyze(
        scenarios=[{"name": "s", "probability": 0.7, "confidence": 0.6}],
        observations=[1, 1, 1, 1, 1], baseline=0.9, samples=1000,
    )
    b = service.analyze(
        scenarios=[{"name": "s", "probability": 0.7, "confidence": 0.6}],
        observations=[0, 0, 0, 0, 0], baseline=0.1, samples=1000,
    )
    assert a.confidence == b.confidence
    assert "confidence_interpretation" in a.semantics
