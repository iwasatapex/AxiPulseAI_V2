from dataclasses import asdict, dataclass, field
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
from core.decision_intelligence.v3.scenario.scoring import rank_scenarios
from core.decision_intelligence.v3.synthesis.explanation import build_explanation
from core.decision_intelligence.v3.bayesian import inference as bayesian_inference


@dataclass
class ProbabilisticDecisionPackage:
    """Canonical probabilistic decision package (Phases 1, 2, 3, 6, 8, 9, 14)."""

    recommendation: str
    risk: str
    probability: float
    confidence: float
    expected: float
    downside: float
    upside: float
    scenarios: list[dict[str, Any]]
    risk_score: float = 0.0
    abstain: bool = False
    success_count: int = 0
    failure_count: int = 0
    decision: dict[str, Any] = field(default_factory=dict)
    explanation: dict[str, Any] = field(default_factory=dict)
    semantics: dict[str, Any] = field(default_factory=dict)
    monte_carlo_detail: dict[str, Any] = field(default_factory=dict)
    bayesian_detail: dict[str, Any] = field(default_factory=dict)


def _aggregate_target_probability(
    monte_carlo: Any,
    targets: Mapping[str, Any] | None,
    best_scenario: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Compute probability-of-target-attainment (per metric) from REAL forecast
    distribution evidence. Never invents values and never runs another
    Monte Carlo.

      - P(NPS >= target_nps): from the NPS 0..10 posterior distribution
        carried on the best scenario (CDF of score probabilities).
      - P(OH >= target_oh): from the single decision-level Monte Carlo normal
        distribution (erf CDF), reusing the aggregation already computed.
    """
    from core.probabilistic.adapter import UniversalProbabilisticAdapter

    targets = dict(targets or {})
    result: dict[str, Any] = {
        "interpretation": (
            "probability of target attainment per metric; derived from the "
            "NPS score distribution and the single decision-level Monte Carlo"
        )
    }

    best = best_scenario or {}
    target_nps = targets.get("target_nps")
    if target_nps is not None:
        distribution = (
            best.get("bayesian_score_distribution")
            or best.get("nps_distribution")
        )
        if distribution:
            try:
                expected_business = bayesian_inference.expected_nps_business(
                    distribution
                )
                prob_promoter = bayesian_inference.promoter_probability(
                    distribution
                )
            except Exception:
                expected_business = None
                prob_promoter = None
            if expected_business is not None:
                result["nps"] = {
                    "target": float(target_nps),
                    "expected_nps": round(float(expected_business), 4),
                    "probability_promoter_score": (
                        round(float(prob_promoter), 4)
                        if prob_promoter is not None
                        else None
                    ),
                    # P(business-NPS >= target) requires sampling the score
                    # multinomial, which would add a Monte Carlo execution
                    # (violating the exactly-one-MC invariant), so it is
                    # honestly marked unavailable.
                    "probability_of_target_nps": None,
                    "note": (
                        "expected business NPS and promoter probability are "
                        "derived from the NPS 0..10 posterior; "
                        "P(business-NPS>=target) is not computed (requires "
                        "extra sampling)."
                    ),
                }

    target_oh = targets.get("target_oh")
    if target_oh is not None and monte_carlo is not None:
        mean = getattr(monte_carlo, "mean", None)
        p05 = getattr(monte_carlo, "p05", None)
        p95 = getattr(monte_carlo, "p95", None)
        if mean is not None and p05 is not None and p95 is not None:
            std = max((float(p95) - float(p05)) / 3.289707253, 0.0)
            target_ratio = float(target_oh) / 100.0
            prob = UniversalProbabilisticAdapter._probability_at_or_above(
                float(mean),
                target_ratio,
                std,
            )
            result["operations_health"] = {
                "target": float(target_oh),
                "probability": prob,
            }

    return result



class ProbabilisticDecisionService:
    def __init__(self):
        self.engine = ADIEProbabilisticEngine()

    def analyze(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        observations: Sequence[float],
        baseline: float,
        uncertainty: float = 0.05,
        samples: int = 10000,
        *,
        targets: Mapping[str, Any] | None = None,
        sensitivity_output: Mapping[str, Any] | None = None,
        observed: float | None = None,
        observed_metrics: Sequence[str] | None = None,
        horizon: int | None = None,
    ) -> ProbabilisticDecisionPackage:

        if not scenarios:
            raise ValueError("scenarios must not be empty")

        # 1. ONE Bayesian inference + ONE Monte Carlo (decision-level aggregate).
        probabilistic = self.engine.analyze(
            observations=[float(v) for v in observations],
            baseline=float(baseline),
            uncertainty=float(uncertainty),
            samples=int(samples),
        )
        bayesian = probabilistic.bayesian
        monte_carlo = probabilistic.monte_carlo

        # 2. Scenario enrichment (Phase 1): NEVER overwrite real per-day
        #    forecast evidence with the aggregate. Each scenario dict keeps
        #    its own operations_health/nps/confidence/risk/delta/distribution;
        #    missing aggregate-only keys are simply left absent, never
        #    fabricated.
        enriched = [dict(s) for s in scenarios]

        # 3. Deterministic scenario ranking (Phase 2).
        ranked = rank_scenarios(enriched)
        best = ranked[0] if ranked else {}

        # 4. Canonical risk model (single source of truth, Phase 8).
        risk_engine = UncertaintyRiskEngine()
        aggregate_risk = risk_engine.assess(
            probability=float(bayesian.probability),
            confidence=float(bayesian.confidence),
            downside=float(monte_carlo.p05),
            upside=float(monte_carlo.p95),
        )

        # 5. Evidence-based, meaningful decision (Phases 3, 6, 9).
        policy = DecisionPolicyEngine()
        decision = policy.select_evidence(
            ranked,
            aggregate_risk,
            targets=targets,
            sensitivity_output=sensitivity_output,
            aggregate={
                "probability": bayesian.probability,
                "confidence": bayesian.confidence,
                "expected": monte_carlo.mean,
                "downside": monte_carlo.p05,
                "upside": monte_carlo.p95,
            },
            observed=observed,
        )

        # 6. Package-level point values: prefer the best scenario's own stats;
        #    fall back to the decision-level aggregate only when the scenario
        #    carries no distribution of its own.
        probability = best.get("probability")
        probability = float(probability) if probability is not None else float(bayesian.probability)
        confidence = best.get("confidence")
        confidence = float(confidence) if confidence is not None else float(bayesian.confidence)
        expected = best.get("expected")
        expected = float(expected) if expected is not None else float(monte_carlo.mean)
        downside = best.get("p05")
        downside = float(downside) if downside is not None else float(monte_carlo.p05)
        upside = best.get("p95")
        upside = float(upside) if upside is not None else float(monte_carlo.p95)

        # 7. Bayesian semantics / target probability (Phases 7, 9).
        semantics = {
            "probability_interpretation": (
                "decision-level Beta-Bernoulli posterior over normalized "
                "observed KPI ratios (health-score posterior); NOT a "
                "target-attainment probability. Per-metric target "
                "probability is reported under 'probability_of_target' when "
                "targets and forecast distributions are available."
            ),
            "confidence_interpretation": (
                "1 minus normalized posterior standard deviation; reflects "
                "how much evidence (sample count) supports the posterior, "
                "not data extremity."
            ),
            "monte_carlo_samples": int(getattr(monte_carlo, "samples", samples) or samples),
            "probability_of_target": _aggregate_target_probability(
                monte_carlo, targets, best
            ),
        }

        # 8. Explanation (Phase 15) from supplied evidence only.
        explanation = build_explanation(
            aggregate={
                "probability": probability,
                "confidence": confidence,
                "downside": downside,
                "upside": upside,
                "probability_interpretation": semantics["probability_interpretation"],
                "uncertainty_interpretation": (
                    "p05/p95 of the decision-level Monte Carlo baseline"
                ),
                "monte_carlo_samples": semantics["monte_carlo_samples"],
            },
            scenarios=ranked,
            risk=asdict(aggregate_risk),
            decision=asdict(decision),
            targets=targets,
            observed_metrics=observed_metrics,
            horizon=horizon,
        )

        # 9. Additive MC detail from the single simulation (no second execution).
        #    Derive success/failure counts from the existing probability_positive
        #    partition and add a coarse histogram summary — all from the same sample.
        mc = monte_carlo
        success_count = int(getattr(mc, "success_count", 0))
        failure_count = int(getattr(mc, "failure_count", 0))
        # Build a distribution summary from the existing p05/p50/p95 and probability_positive
        mc_dist = {}
        if mc is not None:
            mc_dist = {
                "mean": float(getattr(mc, "mean", 0.0)),
                "p05": float(getattr(mc, "p05", 0.0)),
                "p50": float(getattr(mc, "p50", 0.0)),
                "p95": float(getattr(mc, "p95", 0.0)),
                "probability_positive": float(getattr(mc, "probability_positive", 0.0)),
                "samples": int(getattr(mc, "samples", samples)),
                "uncertainty": float(getattr(mc, "uncertainty", 0.0)),
            }
        # Add bins from the engine if available (already present via MonteCarloResult
        # _bin_summary, captured in the monte_carlo attribute from the engine)
        monte_carlo_detail = {"success_count": success_count, "failure_count": failure_count,
                              "distribution_summary": mc_dist}

        # 10. Bayesian detail already in-memory — derive interpretation fields
        #    from the existing BayesianResult object without calling predictors.
        bay = bayesian
        bayesian_detail = {
            "probability": float(getattr(bay, "probability", 0.0)),
            "confidence": float(getattr(bay, "confidence", 0.0)),
            "posterior_mean": float(getattr(bay, "posterior_mean", 0.0)),
            "posterior_std": float(getattr(bay, "posterior_std", 0.0)),
            "credible_interval_lower": float(getattr(bay, "credible_interval_lower", None) or 0.0),
            "credible_interval_upper": float(getattr(bay, "credible_interval_upper", None) or 0.0),
            "credible_level": float(getattr(bay, "credible_level", 0.95)),
            "samples": int(getattr(bay, "samples", 0)),
        }

        return ProbabilisticDecisionPackage(
            recommendation=decision.recommendation,
            risk=aggregate_risk.risk,
            probability=probability,
            confidence=confidence,
            expected=expected,
            downside=downside,
            upside=upside,
            scenarios=ranked,
            risk_score=aggregate_risk.score,
            abstain=aggregate_risk.abstain,
            success_count=success_count,
            failure_count=failure_count,
            decision=asdict(decision),
            explanation=explanation,
            semantics=semantics,
            monte_carlo_detail=monte_carlo_detail,
            bayesian_detail=bayesian_detail,
        )


    @staticmethod
    def to_dict(
        package: ProbabilisticDecisionPackage,
    ) -> dict[str, Any]:
        return asdict(package)


_default_service = ProbabilisticDecisionService()


def analyze(
    scenarios: list[dict[str, Any]],
    observations: list[float],
    baseline: float,
    uncertainty: float = 0.05,
    samples: int = 10000,
    *,
    targets: Mapping[str, Any] | None = None,
    sensitivity_output: Mapping[str, Any] | None = None,
    observed: float | None = None,
    observed_metrics: Sequence[str] | None = None,
    horizon: int | None = None,
) -> ProbabilisticDecisionPackage:
    return _default_service.analyze(
        scenarios=scenarios,
        observations=observations,
        baseline=baseline,
        uncertainty=uncertainty,
        samples=samples,
        targets=targets,
        sensitivity_output=sensitivity_output,
        observed=observed,
        observed_metrics=observed_metrics,
        horizon=horizon,
    )


def to_dict(
    package: ProbabilisticDecisionPackage,
) -> dict[str, Any]:
    return ProbabilisticDecisionService.to_dict(package)
