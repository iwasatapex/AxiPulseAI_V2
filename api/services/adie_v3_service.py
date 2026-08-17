"""
api.services.adie_v3_service

Canonical ADIE V3 service.

This module owns the V3 decision pipeline. It composes the V3
probabilistic engine (Bayesian + Monte Carlo), the risk/uncertainty
engine, the decision policy engine, and the production decision
boundary into a single service used by the V3 API route.

This is the single canonical ADIE decision service. ADIE V2 has been
removed; there is no separate V2 decision path.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.decision_intelligence.v3.intelligence import (
    ADIEProbabilisticEngine,
)
from core.decision_intelligence.v3.risk.uncertainty import (
    UncertaintyRiskEngine,
)
from core.decision_intelligence.v3.policy.decision_policy import (
    DecisionPolicyEngine,
)
from core.decision_intelligence.v3.integration.probabilistic_decision import (
    ProbabilisticDecisionService,
    ProbabilisticDecisionPackage,
)
from core.decision_intelligence.v3.integration.production_boundary import (
    ProductionDecisionBoundary,
)
from core.decision_intelligence.v3.integration.decision_composer import (
    compose_decision_package,
)


class ADIEV3Service:
    """
    Canonical V3 ADIE service.

    Exposes the V3 probabilistic decision pipeline with production
    boundary enforcement. All results are advisory only.
    """

    def __init__(
        self,
        probabilistic_engine: ADIEProbabilisticEngine | None = None,
        risk_engine: UncertaintyRiskEngine | None = None,
        policy_engine: DecisionPolicyEngine | None = None,
        decision_service: ProbabilisticDecisionService | None = None,
        boundary: ProductionDecisionBoundary | None = None,
    ) -> None:
        self.probabilistic = probabilistic_engine or ADIEProbabilisticEngine()
        self.risk_engine = risk_engine or UncertaintyRiskEngine()
        self.policy_engine = policy_engine or DecisionPolicyEngine()
        self.decision_service = decision_service or ProbabilisticDecisionService()
        self.boundary = boundary or ProductionDecisionBoundary(
            decision_service=self.decision_service
        )

    def analyze(
        self,
        observations: Sequence[float],
        baseline: float,
        uncertainty: float = 0.05,
        samples: int = 10000,
        *,
        cutoff: Any = None,
        metadata: Mapping[str, Any] | None = None,
        targets: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Run the V3 probabilistic decision pipeline and return a dict.

        (Bayesian + Monte Carlo -> canonical risk assessment -> evidence-based
        policy decision) with the production decision boundary enforced before
        any decision is produced. This is the same decision semantic model the
        forecast path uses (Phase 6).
        """
        # Enforce the production gate on T inputs BEFORE any probabilistic
        # computation, so that future observed outcomes or predicted
        # recursive state can never reach the engine.
        self.boundary.validate(
            observations=observations,
            baseline=baseline,
            scenarios=None,
            cutoff=cutoff,
            metadata=metadata,
        )

        probabilistic = self.probabilistic.analyze(
            observations=[float(v) for v in observations],
            baseline=float(baseline),
            uncertainty=float(uncertainty),
            samples=int(samples),
        )

        # A single aggregate scenario (the plain path has no per-day forecast
        # evidence). canonical risk + evidence-based decision.
        density = {
            "name": "current_state",
            "probability": probabilistic.bayesian.probability,
            "confidence": probabilistic.bayesian.confidence,
            "expected": probabilistic.monte_carlo.mean,
            "p05": probabilistic.monte_carlo.p05,
            "p95": probabilistic.monte_carlo.p95,
        }

        risk = self.risk_engine.assess(
            probability=probabilistic.bayesian.probability,
            confidence=probabilistic.bayesian.confidence,
            downside=probabilistic.monte_carlo.p05,
            upside=probabilistic.monte_carlo.p95,
        )

        decision = self.policy_engine.select_evidence(
            [density],
            risk,
            targets=targets,
            aggregate=density,
            observed=None,
        )

        explanation = {
            "current_state": {
                "aggregate_probability": probabilistic.bayesian.probability,
                "aggregate_confidence": probabilistic.bayesian.confidence,
                "probability_interpretation": (
                    "decision-level Beta-Bernoulli posterior over normalized "
                    "observed KPI ratios (health-score posterior); NOT a "
                    "target-attainment probability"
                ),
            },
            "uncertainty": {
                "downside": risk.downside,
                "upside": risk.upside,
                "samples": int(samples),
            },
            "main_risk": {
                "level": risk.risk,
                "score": risk.score,
                "abstain": risk.abstain,
            },
            "recommended_action": decision.__dict__,
            "supporting_evidence": decision.evidence,
        }

        return {
            "probabilistic": {
                "bayesian": {
                    "probability": probabilistic.bayesian.probability,
                    "confidence": probabilistic.bayesian.confidence,
                    "posterior_mean": getattr(
                        probabilistic.bayesian, "posterior_mean", None
                    ),
                    "posterior_std": getattr(
                        probabilistic.bayesian, "posterior_std", None
                    ),
                },
                "monte_carlo": {
                    "mean": probabilistic.monte_carlo.mean,
                    "p05": probabilistic.monte_carlo.p05,
                    "p95": probabilistic.monte_carlo.p95,
                },
            },
            "risk": risk.__dict__,
            "decision": decision.__dict__,
            "explanation": explanation,
        }


    def analyze_scenarios(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        observations: Sequence[float],
        baseline: float,
        *,
        uncertainty: float = 0.05,
        samples: int = 10000,
        cutoff: Any = None,
        metadata: Mapping[str, Any] | None = None,
        targets: Mapping[str, Any] | None = None,
        sensitivity_output: Mapping[str, Any] | None = None,
        observed: float | None = None,
        observed_metrics: Sequence[str] | None = None,
        horizon: int | None = None,
    ) -> ProbabilisticDecisionPackage:
        """Run the V3 scenario-based decision service.

        Enforces the production decision boundary (non-empty scenarios,
        finite inputs, temporal provenance) before delegating to
        ProbabilisticDecisionService — the SAME decision semantic model the
        forecast path uses (Phase 6).
        """
        # Enforce the production gate BEFORE any probabilistic computation.
        self.boundary.validate(
            observations=observations,
            baseline=baseline,
            scenarios=scenarios,
            cutoff=cutoff,
            metadata=metadata,
        )

        return self.decision_service.analyze(
            scenarios=[dict(item) for item in scenarios],
            observations=[float(v) for v in observations],
            baseline=float(baseline),
            uncertainty=float(uncertainty),
            samples=int(samples),
            targets=targets,
            sensitivity_output=sensitivity_output,
            observed=observed,
            observed_metrics=observed_metrics,
            horizon=horizon,
        )

    def compose_decision(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        observations: Sequence[float],
        baseline: float,
        *,
        uncertainty: float = 0.05,
        samples: int = 10000,
        cutoff: Any = None,
        metadata: Mapping[str, Any] | None = None,
        recommendation_output: Mapping[str, Any] | None = None,
        strategy_output: Mapping[str, Any] | None = None,
        trend_output: Mapping[str, Any] | None = None,
        sensitivity_output: Mapping[str, Any] | None = None,
        agreement: Mapping[str, Any] | None = None,
        targets: Mapping[str, Any] | None = None,
        observed: float | None = None,
        observed_metrics: Sequence[str] | None = None,
        horizon: int | None = None,
    ) -> dict[str, Any]:
        """Run the canonical V3 decision pipeline and fold Forecast AI
        recommendation/strategy/trend/sensitivity outputs into ONE decision
        payload.

        ADIE V3 owns the decision output contract; the Forecast AI engines
        remain the producers. No data is fabricated — each section is
        included only when supplied.
        """
        package = self.analyze_scenarios(
            scenarios=scenarios,
            observations=observations,
            baseline=baseline,
            uncertainty=uncertainty,
            samples=samples,
            cutoff=cutoff,
            metadata=metadata,
            targets=targets,
            sensitivity_output=sensitivity_output,
            observed=observed,
            observed_metrics=observed_metrics,
            horizon=horizon,
        )

        return compose_decision_package(
            ProbabilisticDecisionService.to_dict(package),
            recommendation_output=recommendation_output,
            strategy_output=strategy_output,
            trend_output=trend_output,
            sensitivity_output=sensitivity_output,
            agreement=agreement,
        )


# Module-level compatibility surface
DEFAULT_SERVICE = ADIEV3Service()


def analyze(
    observations: Sequence[float],
    baseline: float,
    uncertainty: float = 0.05,
    samples: int = 10000,
    *,
    cutoff: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    return DEFAULT_SERVICE.analyze(
        observations=observations,
        baseline=baseline,
        uncertainty=uncertainty,
        samples=samples,
        cutoff=cutoff,
        metadata=metadata,
    )


def analyze_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    observations: Sequence[float],
    baseline: float,
    *,
    uncertainty: float = 0.05,
    samples: int = 10000,
    cutoff: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProbabilisticDecisionPackage:
    return DEFAULT_SERVICE.analyze_scenarios(
        scenarios=scenarios,
        observations=observations,
        baseline=baseline,
        uncertainty=uncertainty,
        samples=samples,
        cutoff=cutoff,
        metadata=metadata,
    )
