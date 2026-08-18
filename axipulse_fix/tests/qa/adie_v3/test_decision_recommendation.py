"""Meaningful, explainable decisions (Phase 3)."""

from core.decision_intelligence.v3.policy.decision_policy import (
    DecisionPolicyEngine,
    PolicyDecision,
)
from core.decision_intelligence.v3.risk.uncertainty import (
    RiskAssessment,
    UncertaintyRiskEngine,
)
from core.decision_intelligence.v3.integration.probabilistic_decision import (
    ProbabilisticDecisionService,
)


def _risk(level, probability, confidence, downside=0.4, upside=0.9):
    return RiskAssessment(
        risk=level, score=0.5, confidence=confidence,
        downside=downside, upside=upside, abstain=False,
    )


def test_recommendation_is_not_hardcoded_across_inputs():
    engine = DecisionPolicyEngine()
    # LOW risk, healthy best day -> maintain (no improvement)
    low = engine.select_evidence(
        [{"name": "forecast_day_1", "operations_health": 82.0, "delta_oh": 0.0}],
        _risk("LOW", 0.85, 0.8),
        observed=82.0,
    )
    # HIGH risk -> monitor
    high = engine.select_evidence(
        [{"name": "forecast_day_1", "operations_health": 82.0}],
        _risk("HIGH", 0.2, 0.3),
        observed=82.0,
    )
    # Abstain -> defer
    abstain = engine.select_evidence(
        [],
        RiskAssessment(risk="HIGH", score=0.9, confidence=0.1, downside=0.1, upside=0.2, abstain=True),
    )
    assert low.recommendation == "maintain_current_plan"
    assert high.recommendation == "monitor_high_risk_forecast"
    assert abstain.recommendation == "defer_action_due_to_uncertainty"
    assert len({low.recommendation, high.recommendation, abstain.recommendation}) == 3


def test_kpi_gap_drives_prioritize_recommendation():
    engine = DecisionPolicyEngine()
    decision = engine.select_evidence(
        [{"name": "forecast_day_1", "operations_health": 82.0, "nps": 60.0}],
        _risk("MEDIUM", 0.6, 0.6),
        targets={"target_oh": 90.0, "target_nps": 75.0},
        observed=82.0,
    )
    assert decision.recommendation == "prioritize_nps_improvement"
    assert decision.affected_kpi == "nps"
    assert decision.reason


def test_sensitivity_drives_prioritize_recommendation():
    engine = DecisionPolicyEngine()
    decision = engine.select_evidence(
        [{"name": "forecast_day_1", "operations_health": 82.0}],
        _risk("MEDIUM", 0.6, 0.6),
        sensitivity_output={
            "ranking": [
                {"metric": "release", "sensitivity_score_oh": 2.5},
                {"metric": "quality", "sensitivity_score_oh": 1.2},
            ]
        },
        observed=82.0,
    )
    assert decision.recommendation == "prioritize_release_improvement"
    assert decision.affected_kpi == "release"


def test_transfer_recommendation_uses_reduction_direction():
    engine = DecisionPolicyEngine()
    decision = engine.select_evidence(
        [{"name": "forecast_day_1", "operations_health": 82.0}],
        _risk("MEDIUM", 0.6, 0.6),
        sensitivity_output={"ranking": [{"metric": "transfer", "sensitivity_score_oh": 3.0}]},
        observed=82.0,
    )
    assert decision.recommendation == "prioritize_transfer_reduction"
    assert decision.direction == "reduce"


def test_decision_is_explainable():
    service = ProbabilisticDecisionService()
    result = service.analyze(
        scenarios=[
            {"name": "current_state", "probability": 0.6, "confidence": 0.6,
             "expected": 0.8, "p05": 0.7, "p95": 0.9},
        ],
        observations=[1, 0, 1],
        baseline=0.7,
        samples=1000,
    )
    assert result.decision.get("reason")
    assert isinstance(result.decision.get("evidence", []), list)
    assert result.explanation.get("preferred_scenario")
    assert result.explanation.get("main_risk")
    assert result.explanation.get("recommended_action")
