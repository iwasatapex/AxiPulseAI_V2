"""
Regression tests for the ADIE recommendation-evidence gap.

Guarantees:
1. A forecast state that should produce a recommendation yields genuine
   recommendation evidence that satisfies the ADIE evidence gate.
2. A forecast state with no actionable policy keeps evidence absent and
   preserves ABSTAIN/insufficient-evidence behaviour.
3. Recommendation evidence (the flat canonical list) reaches decision_detail.
4. Agreement/consistency computes from genuine recommendation evidence.
5. Insufficient evidence still causes ABSTAIN.
6. A forecast-day ranking alone does NOT masquerade as recommendation evidence.
7. No fake conflict_count is introduced on missing evidence.

Root cause regression: the Forecast AI recommendation engine produced a NESTED
payload (``recommendations.recommendations``), but the ADIE evidence gate and
detail builder consume a FLAT list (``recommendations``). Genuine evidence was
therefore discarded before reaching decision_detail. The orchestrator now
normalizes to the flat canonical shape.
"""
from __future__ import annotations

from dataclasses import asdict

from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator as FO
from core.forecast_ai.recommendations import (
    Recommendation as Rec,
    Category as RecCategory,
    Priority as RecPriority,
    Difficulty as RecDifficulty,
)
from core.decision_intelligence.v3.synthesis.decision_detail import (
    DECISION_STATUS_INSUFFICIENT,
    build_adie_detail,
    decision_evidence_sufficient,
)


def _real_rec():
    """A genuine engine-produced recommendation (asdict keeps enum members)."""
    return Rec(
        id="rec-1",
        title="Improve NPS",
        description="Boost NPS toward target",
        category=RecCategory.CUSTOMER_EXPERIENCE,
        priority=RecPriority.HIGH,
        difficulty=RecDifficulty.MEDIUM,
        confidence=0.8,
        actions=["Launch retention campaign"],
        reasoning="nps below target",
        optimization_score=5.0,
        target_kpi="nps",
        direction="increase",
        magnitude=2.0,
    )


def _nested_payload(recs):
    """The shape the engine returned before the fix (nested list)."""
    return {
        "optimization": {"success": bool(recs)},
        "recommendations": {
            "success": bool(recs),
            "recommendations": [asdict(r) for r in recs],
            "warnings": [],
            "errors": [],
            "metadata": {},
        },
    }


def _agreement():
    return {"score": 0.8, "category_consistency": 1.0, "conflicts": []}


def _package(scenarios=None):
    return {
        "scenarios": scenarios or [{"name": "forecast_day_1", "operations_health": 88.0}],
        "semantics": {},
        "decision": None,
    }


# --------------------------------------------------------------------------- #
# 1. Forecast state that should produce a recommendation -> evidence exists
# --------------------------------------------------------------------------- #
def test_actionable_state_yields_genuine_evidence():
    nested = _nested_payload([_real_rec()])
    flat = FO._normalize_recommendation_payload(nested)
    assert flat["status"] == "success"
    assert flat["success"] is True
    assert len(flat["recommendations"]) == 1
    # The recommendation carries enough for agreement/consistency.
    r = flat["recommendations"][0]
    assert r["category"] == "customer_experience"
    assert r["priority"] == "high"
    assert r["confidence"] == 0.8
    assert r["reasoning"]
    assert r["target_kpi"] == "nps"


# --------------------------------------------------------------------------- #
# 2. No actionable policy -> evidence remains absent (ABSTAIN preserved)
# --------------------------------------------------------------------------- #
def test_no_actionable_policy_keeps_evidence_absent():
    flat = FO._normalize_recommendation_payload(_nested_payload([]))
    assert flat["status"] == "skipped"
    assert flat["recommendations"] == []
    sufficient, reason = decision_evidence_sufficient(flat, None)
    assert sufficient is False
    assert reason


# --------------------------------------------------------------------------- #
# 3. Recommendation evidence reaches decision_detail
# --------------------------------------------------------------------------- #
def test_recommendation_evidence_reaches_decision_detail():
    flat = FO._normalize_recommendation_payload(_nested_payload([_real_rec()]))
    detail = build_adie_detail(_package(), recommendation_output=flat,
                               agreement=_agreement(), horizon=5)
    assert detail["decision_status"] == "available"
    assert detail["recommendation_status"] == "available"
    assert len(detail["recommendations"]) >= 1
    assert detail["agreement"]["status"] == "available"


# --------------------------------------------------------------------------- #
# 4. Agreement computes from genuine evidence
# --------------------------------------------------------------------------- #
def test_agreement_computes_from_genuine_evidence():
    flat = FO._normalize_recommendation_payload(_nested_payload([_real_rec()]))
    detail = build_adie_detail(_package(), recommendation_output=flat,
                               agreement=_agreement(), horizon=5)
    ag = detail["agreement"]
    assert ag["available"] is True
    assert ag["status"] == "available"
    assert ag["score"] is not None
    assert ag["category_consistency"] is not None


# --------------------------------------------------------------------------- #
# 5. Insufficient evidence still causes ABSTAIN
# --------------------------------------------------------------------------- #
def test_insufficient_evidence_still_abstains():
    detail = build_adie_detail(_package(), recommendation_output=None,
                               agreement=None, horizon=5)
    assert detail["decision_status"] == DECISION_STATUS_INSUFFICIENT
    assert detail["risk_detail"]["level"] == "ABSTAIN"
    assert detail["risk_detail"]["abstain"] is True
    assert detail["recommendations"] == []


# --------------------------------------------------------------------------- #
# 6. Forecast-day ranking alone is NOT recommendation evidence
# --------------------------------------------------------------------------- #
def test_forecast_ranking_alone_not_recommendation_evidence():
    # Only scenarios (forecast ranking) present; no recommendation_output.
    detail = build_adie_detail(_package(), recommendation_output=None,
                               agreement=None, horizon=5)
    assert detail["decision_status"] == DECISION_STATUS_INSUFFICIENT
    assert detail["scenario_ranking"]["status"] == "forecast_ranking_only"
    assert detail["scenario_ranking"]["actionable"] is False
    # A forecast_day_N ranking alone is not recommendation evidence.
    assert detail["recommendations"] == []


# --------------------------------------------------------------------------- #
# 7. No fake conflict_count on missing evidence
# --------------------------------------------------------------------------- #
def test_no_fake_conflict_count_when_evidence_missing():
    detail = build_adie_detail(_package(), recommendation_output=None,
                               agreement=None, horizon=5)
    ag = detail["agreement"]
    assert ag["available"] is False
    assert ag["status"] == DECISION_STATUS_INSUFFICIENT
    assert ag.get("score") is None
    assert "conflict_count" not in ag


# --------------------------------------------------------------------------- #
# Diagnostics present
# --------------------------------------------------------------------------- #
def test_diagnostics_present():
    flat = FO._normalize_recommendation_payload(_nested_payload([_real_rec()]))
    diag = flat.get("diagnostics", {})
    assert diag.get("engines_evaluated") is True
    assert diag.get("evidence_count") == 1
    assert diag.get("final_recommendation_count") == 1


# --------------------------------------------------------------------------- #
# Canonical shape: top-level counts + single recommendations field
# --------------------------------------------------------------------------- #
def test_canonical_shape_has_single_recommendations_field_and_counts():
    flat = FO._normalize_recommendation_payload(_nested_payload([_real_rec()]))
    # ONE canonical list field — no parallel nested/evidence/field aliases.
    assert isinstance(flat.get("recommendations"), list)
    assert flat.get("recommendations")
    assert flat.get("evidence_count") == 1
    assert flat.get("final_recommendation_count") == 1
    # No separate nested representation is introduced.
    assert not isinstance(flat.get("recommendations"), dict)


# --------------------------------------------------------------------------- #
# Invariant: genuine evidence + agreement => gate MUST see evidence
# --------------------------------------------------------------------------- #
def test_invariant_genuine_evidence_plus_agreement_passes_gate():
    """If len(canonical_recommendations)>0 and agreement is real, the gate MUST
    report recommendation evidence present (never 'Recommendation evidence is
    empty')."""
    flat = FO._normalize_recommendation_payload(_nested_payload([_real_rec()]))
    agreement = {"score": 0.8, "category_consistency": 1.0, "conflicts": []}
    assert len(flat.get("recommendations", [])) > 0
    assert agreement["score"] is not None
    assert agreement["category_consistency"] is not None
    sufficient, reason = decision_evidence_sufficient(flat, agreement)
    assert sufficient is True
    assert "Recommendation evidence is empty" not in reason


# --------------------------------------------------------------------------- #
# Inverse: empty canonical recommendations => insufficient/ABSTAIN preserved
# --------------------------------------------------------------------------- #
def test_inverse_empty_recommendations_keeps_abstain():
    """If canonical_recommendations == [] then agreement may be unavailable and
    the decision must remain insufficient_evidence with ABSTAIN risk."""
    flat = FO._normalize_recommendation_payload(_nested_payload([]))
    assert flat.get("recommendations") == []
    detail = build_adie_detail(_package(), recommendation_output=flat,
                               agreement=None, horizon=5)
    assert detail["decision_status"] == DECISION_STATUS_INSUFFICIENT
    assert detail["risk_detail"]["level"] == "ABSTAIN"
    assert detail["risk_detail"]["abstain"] is True
    assert detail["recommendations"] == []


# --------------------------------------------------------------------------- #
# Exact observed contradiction regression
# --------------------------------------------------------------------------- #
def test_exact_observed_contradiction_resolved():
    """Regression for the reported state: agreement_score=1.0,
    category_consistency=1.0, conflict_count=0, but the gate previously reported
    'Recommendation evidence is empty' / insufficient_evidence / ABSTAIN.

    With genuine recommendation evidence present, the canonical decision must
    become AVAILABLE — the agreement and the gate must see the SAME evidence.
    """
    nested = _nested_payload([_real_rec()])
    # Agreement engine's view (score 1.0 / consistency 1.0 / conflicts 0):
    agreement = {"score": 1.0, "category_consistency": 1.0, "conflicts": [], "conflict_count": 0}
    # The gate must consume the SAME evidence, not a different field:
    sufficient, reason = decision_evidence_sufficient(nested, agreement)
    assert sufficient is True, reason
    assert "Recommendation evidence is empty" not in reason

    detail = build_adie_detail(_package(), recommendation_output=nested,
                               agreement=agreement, horizon=5)
    assert detail["decision_status"] == "available"
    assert detail["recommendation_status"] == "available"
    assert len(detail["recommendations"]) >= 1
    assert detail["agreement"]["score"] == 1.0
    assert detail["agreement"]["status"] == "available"
    assert detail["risk_detail"]["level"] != "ABSTAIN"


# --------------------------------------------------------------------------- #
# Full composer path: canonical decision becomes available with evidence
# --------------------------------------------------------------------------- #
def test_composer_available_when_genuine_evidence_and_agreement():
    from core.decision_intelligence.v3.integration.decision_composer import (
        compose_decision_package,
    )
    nested = _nested_payload([_real_rec()])
    agreement = FO._compute_agreement(nested)
    assert agreement is not None
    prob = {
        "scenarios": [{"name": "forecast_day_1", "operations_health": 88.0}],
        "semantics": {},
        "decision": None,
        "risk": "LOW", "abstain": False,
        "probability": 0.6, "confidence": 0.7,
    }
    pkg = compose_decision_package(prob, recommendation_output=nested,
                                   agreement=agreement,
                                   targets={"target_nps": 75.0}, horizon=5)
    assert pkg["decision_status"] == "available"
    assert pkg["recommendation_status"] == "available"
    assert pkg["probabilistic"]["risk"] != "ABSTAIN"
    assert len(pkg["details"]["recommendations"]) >= 1
