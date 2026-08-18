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


def test_sensitivity_detail_direction_uses_raw_derivative():
    from core.decision_intelligence.v3.synthesis.decision_detail import (
        _build_sensitivity_detail,
    )

    detail = _build_sensitivity_detail(
        {
            "analyses": [
                {
                    "metric": "competency",
                    "oh_change": -0.1524,
                    "sensitivity_score_oh": 0.6988,
                    "sensitivity_score_nps": 0.0,
                },
                {
                    "metric": "transfer",
                    "oh_change": 0.1962,
                    "sensitivity_score_oh": -0.2109,
                    "sensitivity_score_nps": -0.805,
                },
                {
                    "metric": "attendance",
                    "oh_change": -0.1227,
                    "sensitivity_score_oh": -0.0718,
                    "sensitivity_score_nps": 0.0,
                },
            ],
            "ranking": [],
        }
    )

    by_metric = {item["metric"]: item for item in detail["metrics"]}

    # Raw model derivative says increasing competency improves OH.
    assert by_metric["competency"]["direction"] == "increase"
    assert by_metric["competency"]["improvement_direction"] == "increase"
    assert by_metric["competency"]["model_conflict"] is False

    # Raw model derivative says increasing transfer reduces OH, and
    # operational improvement is also to decrease transfer.
    assert by_metric["transfer"]["direction"] == "decrease"
    assert by_metric["transfer"]["improvement_direction"] == "decrease"
    assert by_metric["transfer"]["model_conflict"] is False

    # Attendance is a genuine model conflict: operational improvement is
    # increase, but the raw derivative says OH decreases.
    assert by_metric["attendance"]["direction"] == "decrease"
    assert by_metric["attendance"]["improvement_direction"] == "increase"
    assert by_metric["attendance"]["model_conflict"] is True
