"""
core.decision_intelligence.v3.integration.decision_composer

Canonical ADIE V3 decision composer.

ADIE V3 owns the single decision/recommendation/strategy/trend output
contract. It consumes:
  - the V3 probabilistic decision package (Bayesian + Monte Carlo
    -> risk -> policy),
  - recommendation / strategy / trend outputs produced by the existing
    Forecast AI engines (forecast_ai.engines.recommendation_engine,
    strategy_engine, trend_engine).

The composer does NOT re-implement recommendation/strategy/trend logic:
those engines remain the producers. It only folds their outputs into ONE
canonical decision payload. All inputs are advisory; nothing here
executes a business action and nothing here replaces the trained
predictive models.
"""

from __future__ import annotations

import copy

from typing import Any, Mapping, Sequence

from core.decision_intelligence.v3.synthesis.decision_detail import (
    build_adie_detail,
    decision_evidence_sufficient,
)
from core.decision_intelligence.v3.synthesis.decision_detail import (
    DECISION_STATUS_AVAILABLE,
    DECISION_STATUS_INSUFFICIENT,
)


def compose_decision_package(
    probabilistic: Mapping[str, Any],
    *,
    recommendation_output: Mapping[str, Any] | None = None,
    strategy_output: Mapping[str, Any] | None = None,
    trend_output: Mapping[str, Any] | None = None,
    sensitivity_output: Mapping[str, Any] | None = None,
    agreement: Mapping[str, Any] | None = None,
    targets: Mapping[str, Any] | None = None,
    observed: float | None = None,
    observed_metrics: Sequence[str] | None = None,
    observed_nps: float | None = None,
    observed_state: Mapping[str, Any] | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    """
    Fold Forecast AI recommendation/strategy/trend/sensitivity outputs into
    the canonical V3 decision payload alongside the probabilistic result.

    No data is fabricated: each section is included only when its
    producer output is supplied. The probabilistic package is always
    included (it is the decision core).

    NEW: When called with forecast outputs, also builds the enriched
    ADIE detail via build_adie_detail() and attaches it under 'details'.
    """
    # Keep the producer package intact internally, but expose a canonical
    # decision view that is explicitly ABSTAIN when recommendation/agreement
    # evidence is insufficient. This prevents LOW/MEDIUM raw risk from being
    # mistaken for an actionable decision.
    package: dict[str, Any] = {
        "probabilistic": dict(probabilistic),
    }

    if recommendation_output is not None:
        package["recommendations"] = dict(recommendation_output)
    if strategy_output is not None:
        package["strategies"] = dict(strategy_output)
    if trend_output is not None:
        package["trends"] = dict(trend_output)
    if sensitivity_output is not None:
        package["sensitivity"] = dict(sensitivity_output)
    if agreement is not None:
        package["agreement"] = dict(agreement)

    # Build enriched detail if we have the probabilistic package
    # and at least one forecast output (indicating this is a full forecast->ADIE path)
    has_forecast_output = any([
        recommendation_output,
        strategy_output,
        trend_output,
        sensitivity_output,
        agreement,
    ])
    # Preserve the untouched probabilistic evidence for ADIE detail.
    # The canonical outward decision surface may be gated to ABSTAIN later,
    # but ADIE must retain the underlying Bayesian/Monte Carlo evidence,
    # risk drivers, recommendation evidence, and explanation fields.
    probabilistic_detail_source = copy.deepcopy(probabilistic)

    if has_forecast_output:
        try:
            detail = build_adie_detail(
                probabilistic_detail_source,
                recommendation_output=recommendation_output,
                strategy_output=strategy_output,
                trend_output=trend_output,
                sensitivity_output=sensitivity_output,
                agreement=agreement,
                targets=targets,
                observed=observed,
                observed_metrics=observed_metrics,
                observed_nps=observed_nps,
                observed_state=observed_state,
                horizon=horizon,
            )
            package["details"] = detail
        except Exception as e:
            package["details"] = {"error": f"detail_build_failed: {e}"}

    # --- Evidence gate -------------------------------------------------------
    # The canonical decision/recommendation/risk output is gated on the
    # presence of genuine recommendation AND agreement evidence. When that gate
    # is not met the canonical decision is withheld (risk = ABSTAIN,
    # abstain = True) even though forecast outlook + scenario ranking remain
    # available.
    sufficient, _reason = decision_evidence_sufficient(
        recommendation_output, agreement
    )
    if sufficient:
        package["decision_status"] = DECISION_STATUS_AVAILABLE
        package["recommendation_status"] = DECISION_STATUS_AVAILABLE
    else:
        package["decision_status"] = DECISION_STATUS_INSUFFICIENT
        package["recommendation_status"] = DECISION_STATUS_INSUFFICIENT

        # Canonical outward decision must abstain when recommendation and
        # agreement evidence are insufficient. Preserve the producer's raw
        # Bayesian/Monte-Carlo evidence in ``details``; only the canonical
        # decision surface is gated.
        gated = dict(package["probabilistic"])
        gated["recommendation"] = ""
        gated["recommendation_status"] = DECISION_STATUS_INSUFFICIENT
        gated["risk"] = "ABSTAIN"
        gated["abstain"] = True

        decision = dict(gated.get("decision") or {})
        decision["recommendation"] = ""
        decision["recommendation_status"] = DECISION_STATUS_INSUFFICIENT
        decision["risk"] = "ABSTAIN"
        decision["abstain"] = True
        decision["action"] = "abstain"
        decision["priority"] = "LOW"
        decision["direction"] = "defer"
        decision["reason"] = (
            "recommendation/agreement evidence is insufficient; "
            "canonical decision withheld"
        )
        gated["decision"] = decision
        package["probabilistic"] = gated

    return package


__all__ = ["compose_decision_package"]
