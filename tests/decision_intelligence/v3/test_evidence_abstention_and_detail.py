from __future__ import annotations

from core.decision_intelligence.v3.policy.decision_policy import DecisionPolicyEngine
from core.decision_intelligence.v3.risk.uncertainty import RiskAssessment
from core.decision_intelligence.v3.synthesis.decision_detail import (
    _build_risk_detail,
    _enhance_explanation,
)


def _risk() -> RiskAssessment:
    return RiskAssessment(
        risk="MEDIUM",
        score=0.30,
        confidence=0.80,
        downside=75.0,
        upside=90.0,
        abstain=False,
    )


def test_no_target_objective_abstains_in_policy():
    scenarios = [{"name": "forecast_day_0", "probability": 0.8, "operations_health": 82.0}]
    decision = DecisionPolicyEngine().select_evidence(
        scenarios,
        _risk(),
        targets=None,
        sensitivity_output={
            "ranking": [{"metric": "attendance", "sensitivity_score_oh": 0.9}]
        },
        observed=80.0,
    )
    assert decision.recommendation == ""
    assert decision.risk == "ABSTAIN"
    assert decision.abstain is True


def test_risk_detail_preserves_raw_level_but_canonical_abstains():
    detail = _build_risk_detail(
        {
            "risk": "MEDIUM",
            "risk_score": 0.3,
            "confidence": 0.8,
            "downside": 75.0,
            "upside": 90.0,
        },
        {},
        evidence_sufficient=False,
        evidence_reason="missing recommendation evidence",
    )
    assert detail["level"] == "ABSTAIN"
    assert detail["canonical"]["level"] == "ABSTAIN"
    assert detail["canonical"]["abstain"] is True
    assert detail["status"] == "insufficient_evidence"


def test_explanation_preserves_existing_main_risk_fields():
    result = _enhance_explanation(
        {
            "main_risk": {"driver": "attendance decline", "level": "MEDIUM"},
        },
        {},
        {},
        {},
        evidence_sufficient=False,
        evidence_reason="missing recommendation evidence",
        preferred_name="forecast_day_0",
    )
    assert result["main_risk"]["driver"] == "attendance decline"
    assert result["main_risk"]["level"] == "ABSTAIN"
    assert result["main_risk"]["canonical_level"] == "ABSTAIN"
    assert result["main_risk"]["abstain"] is True
    assert result["decision_status"] == "insufficient_evidence"
