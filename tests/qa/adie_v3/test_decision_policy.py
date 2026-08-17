from dataclasses import dataclass

from core.decision_intelligence.v3.policy.decision_policy import (
    DecisionPolicyEngine,
)
from core.decision_intelligence.v3.risk.uncertainty import (
    RiskAssessment,
)


@dataclass
class Scenario:
    name: str
    probability: float
    confidence: float


def test_execute_low_risk():
    engine = DecisionPolicyEngine()

    risk = RiskAssessment(
        risk="LOW",
        score=0.10,
        confidence=0.75,
        downside=0.80,
        upside=0.95,
        abstain=False,
    )

    result = engine.select(
        [
            Scenario(
                "improved_operations",
                0.8571,
                0.7526,
            )
        ],
        risk,
    )

    assert result.recommendation == "improved_operations"
    assert result.action == "execute"
    assert result.abstain is False


def test_escalate_high_risk():
    engine = DecisionPolicyEngine()

    risk = RiskAssessment(
        risk="HIGH",
        score=0.80,
        confidence=0.30,
        downside=0.20,
        upside=0.60,
        abstain=False,
    )

    result = engine.select(
        [Scenario("current_state", 0.70, 0.30)],
        risk,
    )

    assert result.action == "escalate"
    assert result.priority == "HIGH"


def test_abstain():
    engine = DecisionPolicyEngine()

    risk = RiskAssessment(
        risk="HIGH",
        score=0.90,
        confidence=0.20,
        downside=0.20,
        upside=0.30,
        abstain=True,
    )

    result = engine.select(
        [Scenario("stressed_operations", 0.30, 0.20)],
        risk,
    )

    assert result.action == "abstain"
    assert result.abstain is True


def test_empty_scenarios():
    engine = DecisionPolicyEngine()

    risk = RiskAssessment(
        risk="LOW",
        score=0.10,
        confidence=0.75,
        downside=0.70,
        upside=0.90,
        abstain=False,
    )

    result = engine.select([], risk)

    assert result.action == "abstain"
    assert result.abstain is True
