"""
core.decision_intelligence.v3.policy.constants

Single source of truth for every ADIE V3 decision parameter:

  - the canonical risk model (used by ``UncertaintyRiskEngine`` AND the
    scenario-based ``ProbabilisticDecisionService``),
  - the decision-policy action mapping (used by ``DecisionPolicyEngine``),
  - the deterministic scenario-ranking policy (used by
    ``scenario.scoring.rank_scenarios``).

Every threshold and weight is documented. Changing a value here changes the
behavior of every ADIE entry point (API and forecast) consistently, which is
the Phase-8 requirement: there must be exactly ONE risk/decision semantic
model shared by all ADIE paths.

None of these parameters influence the Forecast AI/model layer. ADIE only
consumes Forecast AI outputs; it never feeds anything back.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Canonical risk model (UncertaintyRiskEngine)
# ---------------------------------------------------------------------------
# The risk score is a weighted combination (0..1, higher = riskier):
#
#     score = (1 - probability) * w.probability
#           + (1 - confidence)  * w.confidence
#           + max(0, probability - downside) * w.downside
#
# The risk LEVEL is a threshold table over (probability, confidence).
# ``abstain`` flags decisions the engine refuses to make without better
# evidence.
RISK_SCORE_WEIGHTS: dict[str, float] = {
    "probability": 0.45,  # low target probability raises risk
    "confidence": 0.35,   # low forecast confidence raises risk
    "downside": 0.20,     # probability below the downside tail raises risk
}

RISK_THRESHOLDS: dict[str, float] = {
    # confidence below high_confidence OR probability below high_probability
    # => HIGH risk.
    "high_confidence": 0.35,
    "high_probability": 0.35,
    # confidence below medium_confidence OR probability below
    # medium_probability => MEDIUM risk (otherwise LOW).
    "medium_confidence": 0.60,
    "medium_probability": 0.60,
}

ABSTAIN_THRESHOLDS: dict[str, float] = {
    # confidence below this always abstains.
    "confidence": 0.25,
    # probability below this abstains only when there is no upside
    # (upside <= downside).
    "probability": 0.40,
}

# ---------------------------------------------------------------------------
# 2. Decision policy (DecisionPolicyEngine)
# ---------------------------------------------------------------------------
DECISION_POLICY: dict[str, object] = {
    # action per risk level
    "high_risk_action": "escalate",
    "medium_risk_action": "review",
    "low_risk_action": "execute",
    "abstain_action": "abstain",
    # priority per risk level
    "high_risk_priority": "HIGH",
    "medium_risk_priority": "MEDIUM",
    "abstain_priority": "LOW",
    # on a LOW-risk decision, a best-scenario probability at or above this
    # threshold earns HIGH priority (otherwise MEDIUM).
    "execute_high_priority_probability": 0.80,
    "execute_high_priority_label": "HIGH",
    "execute_medium_priority_label": "MEDIUM",
}

# ---------------------------------------------------------------------------
# 3. Deterministic scenario-ranking policy (scenario.scoring)
# ---------------------------------------------------------------------------
# score = sum(w_i * component_i) / sum(w_i) over the components that are
# actually available for the scenario. Every component is normalized to
# [0, 1]. A scenario with no evidence at all receives DEFAULT_SCORE and keeps
# its input order on ties (stable sort). The ranking is fully deterministic.
SCENARIO_RANKING_WEIGHTS: dict[str, float] = {
    "performance": 0.30,  # normalized KPI performance (OH, else NPS)
    "probability": 0.25,  # scenario-owned target probability (if supplied)
    "confidence": 0.20,   # forecast confidence (if supplied)
    "safety": 0.15,       # 1 - risk_severity (if supplied)
    "momentum": 0.10,     # normalized forecast delta between days
}

# Score assigned when a scenario provides no ranking evidence at all.
SCENARIO_RANKING_DEFAULT_SCORE: float = 0.0

# Component normalization constants.
PERFORMANCE_OH_SCALE: float = 100.0    # operations_health is 0..100
PERFORMANCE_NPS_SHIFT: float = 100.0   # NPS is -100..100 -> shift then divide
PERFORMANCE_NPS_RANGE: float = 200.0
# A day-over-day operations_health move of this many points saturates the
# momentum component (mapped to 0..1 via clip(delta / swing, -0.5, 0.5) + 0.5).
MOMENTUM_FULL_SWING: float = 20.0

# Float comparison tolerance used for tie detection between scores.
SCORE_TIE_EPSILON: float = 1e-12


__all__ = [
    "RISK_SCORE_WEIGHTS",
    "RISK_THRESHOLDS",
    "ABSTAIN_THRESHOLDS",
    "DECISION_POLICY",
    "SCENARIO_RANKING_WEIGHTS",
    "SCENARIO_RANKING_DEFAULT_SCORE",
    "PERFORMANCE_OH_SCALE",
    "PERFORMANCE_NPS_SHIFT",
    "PERFORMANCE_NPS_RANGE",
    "MOMENTUM_FULL_SWING",
    "SCORE_TIE_EPSILON",
]
