"""
core.decision_intelligence.v3.scenario.scoring

Deterministic scenario ranking policy for ADIE V3 (Phases 1 & 2).

Policy (weights centralized in ``policy.constants``):

    score = sum(w_i * component_i) / sum(w_i)   over AVAILABLE components

Every component is normalized to [0, 1] and is only ever derived from real
forecast evidence carried on the scenario dict:

  - ``performance`` : operations_health / 100, else NPS rescaled to [0,1],
    else a scenario-owned ``expected`` value when it is already normalized.
  - ``probability`` : scenario-owned target probability (0..1).
  - ``confidence``  : scenario-owned forecast confidence (0..1).
  - ``safety``      : 1 - risk_severity, where risk_severity is either the
                      scenario's ``risk_severity`` field or the
                      ``risk["overall_risk"]`` of a Forecast AI risk dict.
  - ``momentum``    : normalized day-over-day ``delta_oh`` (0..1 via
                      clip(delta / MOMENTUM_FULL_SWING, -0.5, 0.5) + 0.5).

Nothing here calls a predictor, runs a Monte Carlo, or invents evidence.
A scenario with no evidence receives ``SCENARIO_RANKING_DEFAULT_SCORE``.

Tie behavior: ranking is a stable sort by score descending. Two scenarios
with equal scores retain their input order and are both marked ``tie``
(the later one), i.e. the tie is preserved rather than manufactured away.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from core.decision_intelligence.v3.policy import constants as C


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _risk_severity(scenario: Mapping[str, Any]) -> float | None:
    """Extract a normalized risk severity (0..1) from a scenario, if any."""
    severity = scenario.get("risk_severity")
    if severity is not None and _finite(severity):
        return _clamp(severity)
    risk = scenario.get("risk")
    if isinstance(risk, dict):
        overall = risk.get("overall_risk")
        if overall is not None and _finite(overall):
            return _clamp(overall)
    return None


def _performance_component(scenario: Mapping[str, Any]) -> float | None:
    oh = scenario.get("operations_health")
    if oh is not None and _finite(oh):
        return _clamp(float(oh) / C.PERFORMANCE_OH_SCALE)
    nps = scenario.get("nps")
    if nps is not None and _finite(nps):
        return _clamp(
            (float(nps) + C.PERFORMANCE_NPS_SHIFT) / C.PERFORMANCE_NPS_RANGE
        )
    expected = scenario.get("expected")
    if expected is not None and _finite(expected):
        value = float(expected)
        if 0.0 <= value <= 1.0:
            return value
    return None


def compute_scenario_score(
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Compute the deterministic ranking score for one scenario.

    Returns ``{"score": float, "components": {name: value},
    "available": [names]}``. ``components`` only contains real evidence.
    """
    components: dict[str, float] = {}

    performance = _performance_component(scenario)
    if performance is not None:
        components["performance"] = performance

    probability = scenario.get("probability")
    if probability is not None and _finite(probability):
        components["probability"] = _clamp(float(probability))

    confidence = scenario.get("confidence")
    if confidence is not None and _finite(confidence):
        components["confidence"] = _clamp(float(confidence))

    severity = _risk_severity(scenario)
    if severity is not None:
        components["safety"] = 1.0 - severity

    delta = scenario.get("delta_oh")
    if delta is not None and _finite(delta):
        swing = float(delta) / C.MOMENTUM_FULL_SWING
        swing = max(-0.5, min(0.5, swing))
        components["momentum"] = swing + 0.5

    if not components:
        return {
            "score": C.SCENARIO_RANKING_DEFAULT_SCORE,
            "components": {},
            "available": [],
        }

    weights = C.SCENARIO_RANKING_WEIGHTS
    available = [name for name in components if name in weights]
    total_weight = sum(weights[name] for name in available)

    if total_weight <= 0.0:
        return {
            "score": C.SCENARIO_RANKING_DEFAULT_SCORE,
            "components": components,
            "available": available,
        }

    score = (
        sum(weights[name] * components[name] for name in available)
        / total_weight
    )

    return {
        "score": round(score, 6),
        "components": components,
        "available": available,
    }


def rank_scenarios(
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Rank scenario dicts deterministically and annotate each with
    ``rank``, ``score``, ``tie`` and an ``evidence`` block.

    The input list is mutated (each scenario dict gains the ranking keys) and
    returned in ranked order. Stable sort preserves input order on ties.
    """
    scored = [(compute_scenario_score(s), s) for s in scenarios]

    ranked = sorted(scored, key=lambda pair: -pair[0]["score"])

    previous_score: float | None = None
    result: list[dict[str, Any]] = []
    for index, (score_result, scenario) in enumerate(ranked, start=1):
        score = score_result["score"]
        tie = (
            previous_score is not None
            and abs(previous_score - score) < C.SCORE_TIE_EPSILON
        )
        scenario["rank"] = index
        scenario["score"] = score
        scenario["tie"] = bool(tie)
        scenario["evidence"] = {
            "components": score_result["components"],
            "available": score_result["available"],
            "policy": "sum(w_i * component_i) / sum(w_i) over available "
            "evidence; see policy.constants.SCENARIO_RANKING_WEIGHTS",
        }
        previous_score = score
        result.append(scenario)

    return result


__all__ = [
    "compute_scenario_score",
    "rank_scenarios",
]
