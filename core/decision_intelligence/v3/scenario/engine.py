from dataclasses import dataclass, asdict
from typing import Any

from core.decision_intelligence.v3.intelligence import (
    ADIEProbabilisticEngine,
)


@dataclass
class Scenario:
    name: str
    baseline: float
    uncertainty: float
    observations: list[float]


@dataclass
class ScenarioResult:
    name: str
    probability: float
    confidence: float
    expected_value: float
    p05: float
    p50: float
    p95: float
    probability_positive: float
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ADIEScenarioEngine:

    def __init__(self, samples: int = 10000):
        if samples <= 0:
            raise ValueError("samples must be positive")

        self.samples = samples
        self.probabilistic = ADIEProbabilisticEngine()

    def run(self, scenario: Scenario) -> ScenarioResult:
        result = self.probabilistic.analyze(
            observations=scenario.observations,
            baseline=scenario.baseline,
            uncertainty=scenario.uncertainty,
            samples=self.samples,
        )

        return ScenarioResult(
            name=scenario.name,
            probability=result.bayesian.probability,
            confidence=result.bayesian.confidence,
            expected_value=result.monte_carlo.mean,
            p05=result.monte_carlo.p05,
            p50=result.monte_carlo.p50,
            p95=result.monte_carlo.p95,
            probability_positive=result.monte_carlo.probability_positive,
            samples=result.monte_carlo.samples,
        )

    def compare(
        self,
        scenarios: list[Scenario],
    ) -> list[ScenarioResult]:

        results = [self.run(scenario) for scenario in scenarios]

        return sorted(
            results,
            key=lambda x: (
                x.probability_positive,
                x.probability,
                x.expected_value,
            ),
            reverse=True,
        )


def to_dict(scenario_result: ScenarioResult | None) -> dict:
    """Module-level convenience: serialize a ScenarioResult to a dict."""
    if scenario_result is None:
        return {}
    if hasattr(scenario_result, "to_dict"):
        return scenario_result.to_dict()
    try:
        from dataclasses import asdict
        return asdict(scenario_result)
    except Exception:
        return {k: getattr(scenario_result, k, None) for k in (
            "name", "probability", "confidence", "expected_value",
            "p05", "p50", "p95", "probability_positive", "samples",
        ) if hasattr(scenario_result, k)}


__all__ = ["ADIEScenarioEngine", "Scenario", "ScenarioResult", "to_dict"]


def run(scenario: Scenario) -> ScenarioResult:
    """Module-level convenience: run a single scenario through the engine."""
    return ADIEScenarioEngine().run(scenario)


def add(*scenarios) -> list[dict[str, Any]]:
    """Module-level convenience: collect scenario definitions into a list."""
    return [dict(s) if not isinstance(s, dict) else s for s in scenarios]


def compare(scenarios: list[Scenario]) -> list[ScenarioResult]:
    """Module-level convenience: compare multiple scenarios."""
    return ADIEScenarioEngine().compare(scenarios)


__all__ = ["ADIEScenarioEngine", "Scenario", "ScenarioResult", "to_dict", "run", "add", "compare"]
