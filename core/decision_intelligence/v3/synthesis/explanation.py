"""
core.decision_intelligence.v3.synthesis.explanation

Deterministic, evidence-based decision explanations (Phase 15).

The explanation is built ONLY from supplied evidence — never invented.
It answers, in order:

  1. What is the current state?
  2. What is forecast?
  3. What is the preferred scenario?
  4. Why was it preferred?
  5. What is the uncertainty?
  6. What is the main risk?
  7. What action is recommended?
  8. What evidence supports that action?
  9. What would change the decision?

Every field is derived from the aggregate probabilistic result, the ranked
scenarios, the canonical risk model output, and the decision policy output.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_explanation(
    *,
    aggregate: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    risk: Mapping[str, Any],
    decision: Mapping[str, Any],
    targets: Mapping[str, Any] | None = None,
    observed_metrics: Sequence[str] | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    """
    Build a structured explanation from decision evidence.

    ``aggregate``  : the decision-level Bayesian/Monte Carlo summary.
    ``scenarios``  : the ranked scenario list (rank 1 = preferred).
    ``risk``       : the canonical RiskAssessment as a dict.
    ``decision``   : the DecisionPolicyEngine output as a dict.
    """
    targets = dict(targets or {})
    observed_metrics = list(observed_metrics or [])

    best = scenarios[0] if scenarios else None

    # 1. Current state.
    current_state = {
        "aggregate_probability": aggregate.get("probability"),
        "aggregate_confidence": aggregate.get("confidence"),
        "observed_metrics": observed_metrics,
        "probability_interpretation": aggregate.get(
            "probability_interpretation",
            "Beta-Bernoulli posterior over normalized observed KPI ratios; "
            "a health-score posterior, NOT a target-attainment probability.",
        ),
    }

    # 2. What is forecast.
    forecast_summary = {
        "scenario_count": len(scenarios),
        "horizon_days": horizon,
        "best_scenario": (best or {}).get("name"),
        "best_score": (best or {}).get("score"),
        "targets": targets or None,
    }

    # 3 / 4. Preferred scenario + why.
    if best is not None:
        preferred_scenario = {
            "name": best.get("name"),
            "rank": best.get("rank"),
            "score": best.get("score"),
            "tie": best.get("tie", False),
            "evidence_components": (best.get("evidence") or {}).get(
                "components"
            ),
            "evidence_available": (best.get("evidence") or {}).get(
                "available"
            ),
            "date": best.get("date"),
        }
        why_preferred = {
            "policy": (best.get("evidence") or {}).get(
                "policy",
                "weighted average of available normalized forecast evidence",
            ),
            "available_components": (best.get("evidence") or {}).get(
                "available"
            ),
            "component_scores": (best.get("evidence") or {}).get(
                "components"
            ),
            "note": (
                "Scenario ranked by the deterministic policy in "
                "policy.constants.SCENARIO_RANKING_WEIGHTS; "
                "no predictor calls, no additional Monte Carlo."
            ),
        }
    else:
        preferred_scenario = None
        why_preferred = {"note": "no scenario evidence available"}

    # 5. Uncertainty.
    uncertainty = {
        "downside": aggregate.get("downside"),
        "upside": aggregate.get("upside"),
        "confidence": aggregate.get("confidence"),
        "monte_carlo_samples": aggregate.get("monte_carlo_samples"),
        "interpretation": aggregate.get(
            "uncertainty_interpretation",
            "downside/upside are the Monte Carlo p05/p95 of the decision-level "
            "baseline distribution",
        ),
    }

    # 6. Main risk.
    main_risk = {
        "level": risk.get("risk"),
        "score": risk.get("score"),
        "confidence": risk.get("confidence"),
        "downside": risk.get("downside"),
        "upside": risk.get("upside"),
        "abstain": risk.get("abstain"),
    }

    # 7 / 8. Recommended action + evidence.
    recommended_action = {
        "recommendation": decision.get("recommendation"),
        "action": decision.get("action"),
        "priority": decision.get("priority"),
        "reason": decision.get("reason"),
        "affected_kpi": decision.get("affected_kpi"),
        "direction": decision.get("direction"),
    }

    # 9. What would change the decision.
    decision_changers = {
        "re_ranking": (
            "a materially different forecast scenario (higher OH/NPS, "
            "confidence, safety, or momentum) would change the preferred "
            "scenario via the deterministic ranking policy"
        ),
        "risk": (
            "a change in aggregate probability or confidence would move the "
            "canonical risk level and therefore the recommended action"
        ),
    }

    return {
        "current_state": current_state,
        "forecast_summary": forecast_summary,
        "preferred_scenario": preferred_scenario,
        "why_preferred": why_preferred,
        "uncertainty": uncertainty,
        "main_risk": main_risk,
        "recommended_action": recommended_action,
        "supporting_evidence": list(decision.get("evidence") or []),
        "decision_changers": decision_changers,
    }


__all__ = ["build_explanation"]
