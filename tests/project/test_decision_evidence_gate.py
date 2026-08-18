"""
Regression tests for the ADIE V2/V3 decision evidence gate.

Guarantees:
- A canonical decision/recommendation/risk is ONLY produced when genuine
  recommendation AND agreement evidence are present.
- When the evidence gate is insufficient:
    * decision_status = "insufficient_evidence"
    * main risk abstain = True, level = "ABSTAIN"
    * recommendations = [] and recommendation_status = "insufficient_evidence"
    * no fake conflict_count = 0 when conflict analysis did not run
    * scenario ranking stays available but is labelled forecast-ranking-only
    * the preferred scenario is an explicit, non-actionable forecast preference
- With sufficient evidence the normal decision/risk flow still works.
- Existing forecast/NPS semantics, model invocation and GUI model-family
  selection are unchanged.
"""
import math

import pytest

from core.decision_intelligence.v3.bayesian import inference as bayes
from core.decision_intelligence.v3.synthesis.decision_detail import (
    DECISION_STATUS_INSUFFICIENT,
    build_adie_detail,
    decision_evidence_sufficient,
)


def _nps_distribution(center=8):
    weights = {s: math.exp(-0.5 * (s - center) ** 2) for s in range(0, 11)}
    total = sum(weights.values())
    return {f"score_{s}": w / total for s, w in weights.items()}


def _day(name, oh, conf, risk_sev, delta, center, rank):
    dist = _nps_distribution(center)
    nps = bayes.expected_nps_business(dist)
    return {
        "name": name,
        "rank": rank,
        "_predicted": True,
        "operations_health": oh,
        "nps": nps,
        "confidence": conf,
        "risk_severity": risk_sev,
        "delta_oh": delta,
        "expected_score": bayes.expected_nps_from_distribution(dist),
        "score_p05": 6.0,
        "score_p95": 10.0,
        "nps_p05": 60.0,
        "nps_p95": 100.0,
        "bayesian_score_distribution": dist,
    }


def _scenarios():
    """Pre-ranked scenarios with forecast_day_1 on top (deterministic ranking)."""
    return [
        _day("forecast_day_1", 92.0, 0.9, 0.1, 2.0, 9, 1),
        _day("forecast_day_2", 88.0, 0.7, 0.3, 0.5, 8, 2),
        _day("forecast_day_0", 80.0, 0.6, 0.5, -1.0, 7, 3),
    ]


def _package(risk="LOW", abstain=False, risk_score=0.21, confidence=0.8, probability=0.85,
             downside=0.7, upside=0.95):
    return {
        "scenarios": _scenarios(),
        "semantics": {},
        "decision": {
            "recommendation": "maintain_current_plan",
            "action": "execute",
            "priority": "MEDIUM",
            "risk": risk,
            "confidence": confidence,
            "abstain": abstain,
            "reason": "test decision",
            "evidence": [],
        },
        "risk": risk,
        "risk_score": risk_score,
        "abstain": abstain,
        "confidence": confidence,
        "probability": probability,
        "downside": downside,
        "upside": upside,
        "explanation": {
            "preferred_scenario": {"name": "forecast_day_1", "rank": 1, "score": 0.9},
            "why_preferred": {"policy": "deterministic"},
            "main_risk": {"level": risk, "score": risk_score, "abstain": abstain},
            "recommended_action": {"recommendation": "maintain", "action": "execute"},
        },
        "bayesian_detail": {},
        "monte_carlo_detail": {},
    }


def _recommendation_output():
    return {
        "status": "success",
        "recommendations": [
            {
                "title": "Improve NPS",
                "target_kpi": "nps",
                "direction": "increase",
                "priority": "high",
                "confidence": 0.8,
                "reasoning": "nps below target",
            },
        ],
    }


def _agreement():
    return {"score": 0.8, "category_consistency": 1.0, "conflicts": []}

# --------------------------------------------------------------------------- #
# 1. Insufficient agreement evidence
# --------------------------------------------------------------------------- #
def test_insufficient_agreement_sets_insufficient_evidence_status():
    detail = build_adie_detail(
        _package(),
        recommendation_output=_recommendation_output(),
        agreement=None,
        horizon=3,
    )
    assert detail["decision_status"] == DECISION_STATUS_INSUFFICIENT
    assert detail["recommendation_status"] == DECISION_STATUS_INSUFFICIENT
    assert detail["agreement"]["status"] == DECISION_STATUS_INSUFFICIENT
    assert detail["agreement"]["available"] is False


# --------------------------------------------------------------------------- #
# 2. Insufficient recommendation evidence -> abstain = true
# --------------------------------------------------------------------------- #
def test_insufficient_recommendation_evidence_abstains():
    detail = build_adie_detail(
        _package(risk="LOW", abstain=False),
        recommendation_output={"status": "skipped", "reason": "missing_target",
                               "recommendations": []},
        agreement=None,
        horizon=3,
    )
    assert detail["decision_status"] == DECISION_STATUS_INSUFFICIENT
    assert detail["recommendations"] == []
    rd = detail["risk_detail"]
    assert rd["level"] == "ABSTAIN"
    assert rd["abstain"] is True
    assert rd["status"] == DECISION_STATUS_INSUFFICIENT
    # Raw score preserved only as diagnostic metadata, not a canonical decision.
    assert rd["raw"]["score"] == pytest.approx(0.21)
    assert rd["score"] is None


# --------------------------------------------------------------------------- #
# 3. No fake conflict_count = 0 when conflict analysis did not run
# --------------------------------------------------------------------------- #
def test_no_fake_conflict_count_zero_when_conflict_not_run():
    # Agreement missing entirely.
    detail = build_adie_detail(
        _package(), recommendation_output=_recommendation_output(), agreement=None, horizon=3,
    )
    ag = detail["agreement"]
    assert ag["status"] == DECISION_STATUS_INSUFFICIENT
    assert "conflict_count" not in ag

    # Agreement present but incomplete (no score) -> still insufficient, no 0.
    detail2 = build_adie_detail(
        _package(),
        recommendation_output=_recommendation_output(),
        agreement={"category_consistency": 1.0, "conflicts": []},
        horizon=3,
    )
    ag2 = detail2["agreement"]
    assert ag2["status"] == DECISION_STATUS_INSUFFICIENT
    assert ag2.get("score") is None
    assert ag2.get("conflict_count") is None

    # The gate helper agrees: incomplete agreement is not sufficient evidence.
    sufficient, _ = decision_evidence_sufficient(_recommendation_output(),
                                                 {"category_consistency": 1.0})
    assert sufficient is False



# --------------------------------------------------------------------------- #
# 4. Scenario ranking still returns forecast_day_1 when evidence insufficient
# --------------------------------------------------------------------------- #
def test_scenario_ranking_still_available_when_insufficient():
    detail = build_adie_detail(
        _package(),
        recommendation_output={"status": "skipped", "reason": "missing_target",
                               "recommendations": []},
        agreement=None,
        horizon=3,
    )
    comp = detail["scenario_comparison"]
    assert len(comp) >= 1
    assert comp[0]["name"] == "forecast_day_1"
    assert comp[0]["rank"] == 1
    # Labelled forecast-ranking-only, not a decision.
    assert detail["scenario_ranking"]["status"] == "forecast_ranking_only"
    assert detail["scenario_ranking"]["actionable"] is False


# --------------------------------------------------------------------------- #
# 5. Forecast preference explicitly labelled non-actionable
# --------------------------------------------------------------------------- #
def test_forecast_preference_explicitly_non_actionable():
    detail = build_adie_detail(
        _package(),
        recommendation_output={"status": "skipped", "reason": "missing_target",
                               "recommendations": []},
        agreement=None,
        horizon=3,
    )
    fp = detail["forecast_preference"]
    assert fp["name"] == "forecast_day_1"
    assert fp["actionable"] is False
    assert fp["kind"] == "forecast_preference"

    exp = detail["explanation"]
    assert exp["forecast_preference"]["actionable"] is False
    assert exp["preferred_scenario"]["actionable"] is False
    assert exp["recommended_action"]["action"] == "withheld"
    assert exp["main_risk"]["level"] == "ABSTAIN"
    assert exp["main_risk"]["abstain"] is True
    assert exp["decision_status"] == DECISION_STATUS_INSUFFICIENT
    # "Why selected" wording is a forecast preference, not an actionable decision.
    ws = exp["why_selected"]["text"]
    assert "Forecast preference only" in ws
    assert "forecast_day_1" in ws
    assert "insufficient" in ws


# --------------------------------------------------------------------------- #
# 6. Sufficient evidence permits normal decision / risk flow
# --------------------------------------------------------------------------- #
def test_sufficient_evidence_permits_normal_flow():
    detail = build_adie_detail(
        _package(risk="LOW", abstain=False),
        recommendation_output=_recommendation_output(),
        agreement=_agreement(),
        horizon=3,
    )
    assert detail["decision_status"] == "available"
    assert detail["recommendation_status"] == "available"
    assert len(detail["recommendations"]) == 1
    assert detail["risk_detail"]["level"] == "LOW"
    assert detail["risk_detail"]["abstain"] is False
    assert detail["explanation"]["decision_status"] == "available"
    assert detail["scenario_ranking"]["actionable"] is True

# --------------------------------------------------------------------------- #
# 7. Recommendations remain empty when evidence unavailable
# --------------------------------------------------------------------------- #
def test_recommendations_empty_when_evidence_unavailable():
    detail = build_adie_detail(
        _package(),
        recommendation_output={"status": "skipped", "reason": "missing_target",
                               "recommendations": []},
        agreement=None,
        horizon=3,
    )
    assert detail["recommendations"] == []
    assert detail["recommendation_status"] == DECISION_STATUS_INSUFFICIENT


# --------------------------------------------------------------------------- #
# 8. Existing forecast / NPS semantics remain unchanged
# --------------------------------------------------------------------------- #
def test_forecast_nps_semantics_unchanged():
    detail = build_adie_detail(
        _package(),
        recommendation_output=_recommendation_output(),
        agreement=_agreement(),
        horizon=3,
    )
    first = detail["forecast_summary"]["per_day_table"][0]
    assert 0.0 <= first["expected_score"] <= 10.0
    assert abs(first["nps"]) <= 100.0
    assert -100.0 <= first["nps_p05"] <= first["nps_p95"] <= 100.0
    assert 0.0 <= first["score_p05"] <= first["score_p95"] <= 10.0


# --------------------------------------------------------------------------- #
# 9. Existing model invocation remains unchanged
# --------------------------------------------------------------------------- #
def test_model_invocation_unchanged():
    import inspect

    from core.forecast_ai.engines import forecast_orchestrator

    src = inspect.getsource(forecast_orchestrator.ForecastOrchestrator.execute)
    assert "self.service.predict(pred_req)" in src


# --------------------------------------------------------------------------- #
# 10. Existing GUI model-family selection remains unchanged
# --------------------------------------------------------------------------- #
def test_gui_model_family_selection_unchanged():
    from core.forecast_ai.prediction.model_selector import list_model_families
    from gui import model_selection as ms

    assert hasattr(ms, "get_feature_selection")
    assert hasattr(ms, "set_feature_selection")
    assert hasattr(ms, "render_model_selector")
    assert isinstance(list_model_families(), list)


# --------------------------------------------------------------------------- #
# Composer-level gating (the canonical decision payload / GUI surface)
# --------------------------------------------------------------------------- #
def test_composer_abstains_when_evidence_insufficient():
    from core.decision_intelligence.v3.integration.decision_composer import (
        compose_decision_package,
    )

    pkg = compose_decision_package(
        _package(risk="LOW", abstain=False),
        recommendation_output={"status": "skipped", "reason": "missing_target",
                               "recommendations": []},
        agreement=None,
        targets={},
        observed=80.0,
        observed_metrics=["operations_health"],
        horizon=3,
    )
    assert pkg["decision_status"] == DECISION_STATUS_INSUFFICIENT
    prob = pkg["probabilistic"]
    assert prob["risk"] == "ABSTAIN"
    assert prob["abstain"] is True
    assert prob["recommendation_status"] == DECISION_STATUS_INSUFFICIENT
    details = pkg["details"]
    assert details["decision_status"] == DECISION_STATUS_INSUFFICIENT
    assert details["risk_detail"]["level"] == "ABSTAIN"
    assert details["risk_detail"]["abstain"] is True
    assert details["recommendations"] == []


def test_composer_normal_flow_when_sufficient():
    from core.decision_intelligence.v3.integration.decision_composer import (
        compose_decision_package,
    )

    pkg = compose_decision_package(
        _package(risk="LOW", abstain=False),
        recommendation_output=_recommendation_output(),
        agreement=_agreement(),
        targets={},
        observed=80.0,
        observed_metrics=["operations_health"],
        horizon=3,
    )
    assert pkg["decision_status"] == "available"
    assert pkg["probabilistic"]["risk"] == "LOW"
    assert pkg["probabilistic"]["abstain"] is False
    assert pkg["details"]["decision_status"] == "available"
    assert len(pkg["details"]["recommendations"]) == 1


# --------------------------------------------------------------------------- #
# P0-A: BEST-EFFORT MUST NEVER BECOME CANONICAL ACTIONABLE
# --------------------------------------------------------------------------- #
def _best_effort_recommendation_output():
    """A genuine advisory recommendation produced when the optimizer preserved
    an improving candidate on timeout (best_effort=true, goal_achieved=false)."""
    return {
        "status": "success",
        "success": True,
        "recommendations": [
            {
                "title": "Raise quality toward target (best-effort)",
                "target_kpi": "quality",
                "direction": "increase",
                "priority": "high",
                "confidence": 0.8,
                "reasoning": "genuine improving candidate preserved on timeout",
                "metadata": {"best_effort": True, "goal_achieved": False},
            },
        ],
        "metadata": {
            "best_effort": True,
            "goal_achieved": False,
            "reason": "best_effort",
        },
    }


def test_best_effort_recommendation_alone_is_not_canonical_actionable():
    """A best-effort advisory recommendation must NOT satisfy the canonical
    actionable evidence gate, even though genuine rec + agreement evidence are
    present. The canonical decision must abstain."""
    sufficient, reason = decision_evidence_sufficient(
        _best_effort_recommendation_output(),
        _agreement(),
    )
    assert sufficient is False
    assert "best-effort" in reason

    detail = build_adie_detail(
        _package(risk="LOW", abstain=False),
        recommendation_output=_best_effort_recommendation_output(),
        agreement=_agreement(),
        horizon=3,
    )
    assert detail["decision_status"] == DECISION_STATUS_INSUFFICIENT
    assert detail["recommendation_status"] == DECISION_STATUS_INSUFFICIENT
    assert detail["risk_detail"]["level"] == "ABSTAIN"
    assert detail["risk_detail"]["abstain"] is True


def test_best_effort_recommendation_preserved_in_detail():
    """The underlying best-effort recommendation evidence must be preserved for
    audit/detail even though the canonical decision abstains. It must remain
    explicitly marked best_effort=true."""
    detail = build_adie_detail(
        _package(risk="LOW", abstain=False),
        recommendation_output=_best_effort_recommendation_output(),
        agreement=_agreement(),
        horizon=3,
    )
    assert detail["decision_status"] == DECISION_STATUS_INSUFFICIENT
    recs = detail["recommendations"]
    assert len(recs) == 1
    assert recs[0]["metadata"].get("best_effort") is True
    assert recs[0]["metadata"].get("goal_achieved") is False
    # The advisory rec is still surfaced (not destroyed).
    assert recs[0]["action"]


def test_composer_abstains_on_best_effort_but_preserves_producer_evidence():
    """Composer-level: a best-effort recommendation yields an ABSTAIN canonical
    decision (risk=ABSTAIN, abstain=true, recommendation="") while the detail
    still exposes the advisory producer evidence."""
    from core.decision_intelligence.v3.integration.decision_composer import (
        compose_decision_package,
    )

    pkg = compose_decision_package(
        _package(risk="LOW", abstain=False),
        recommendation_output=_best_effort_recommendation_output(),
        agreement=_agreement(),
        targets={},
        observed=80.0,
        observed_metrics=["operations_health"],
        horizon=3,
    )
    assert pkg["decision_status"] == DECISION_STATUS_INSUFFICIENT
    prob = pkg["probabilistic"]
    assert prob["risk"] == "ABSTAIN"
    assert prob["abstain"] is True
    assert prob["recommendation_status"] == DECISION_STATUS_INSUFFICIENT
    # Raw producer evidence preserved in details.
    assert len(pkg["details"]["recommendations"]) == 1
    assert pkg["details"]["recommendations"][0]["metadata"].get("best_effort") is True


def test_fully_achieved_recommendation_remains_actionable():
    """A valid fully achieved recommendation (best_effort=false,
    goal_achieved=true) must remain actionable."""
    achieved = {
        "status": "success",
        "success": True,
        "recommendations": [
            {
                "title": "Raise quality",
                "target_kpi": "quality",
                "direction": "increase",
                "priority": "high",
                "confidence": 0.8,
                "metadata": {"best_effort": False, "goal_achieved": True},
            },
        ],
        "metadata": {"best_effort": False, "goal_achieved": True},
    }
    sufficient, _reason = decision_evidence_sufficient(achieved, _agreement())
    assert sufficient is True

    detail = build_adie_detail(
        _package(risk="LOW", abstain=False),
        recommendation_output=achieved,
        agreement=_agreement(),
        horizon=3,
    )
    assert detail["decision_status"] == "available"
    assert detail["risk_detail"]["level"] == "LOW"
    assert detail["risk_detail"]["abstain"] is False

