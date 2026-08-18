from dataclasses import dataclass
from types import SimpleNamespace

from core.probabilistic import (
    ProbabilisticResult,
    UniversalProbabilisticAdapter as CoreProbabilisticAdapter,
)


@dataclass
class PredictorProbabilisticResult:
    predictor: str
    bayesian: object
    monte_carlo: object
    probability: float
    confidence: float
    expected: float
    downside: float
    upside: float
    simulations: int
    uncertainty: float
    historical_samples: int
    result: ProbabilisticResult | None = None


class UniversalProbabilisticAdapter:

    def __init__(self):
        self.core = CoreProbabilisticAdapter()

    @staticmethod
    def _metric_scale(predictor: str) -> float:
        scales = {
            "nps": 100.0,
            "quality": 100.0,
            "competency": 100.0,
            "attendance": 100.0,
            "release": 100.0,
            "transfer": 20.0,
            "operations_health": 100.0,
        }
        return scales.get(predictor, 100.0)

    @classmethod
    def _derive_uncertainty(
        cls,
        predictor: str,
        values: list[float],
        explicit: float | None,
    ) -> float:
        if explicit is not None:
            return max(float(explicit), 0.0)

        if len(values) >= 2:
            import statistics

            spread = statistics.stdev(values)
            return max(
                spread,
                cls._metric_scale(predictor) * 0.01,
            )

        return cls._metric_scale(predictor) * 0.02

    @staticmethod
    def _normalize(
        predictor: str,
        values: list[float],
    ) -> list[float]:
        scale = UniversalProbabilisticAdapter._metric_scale(predictor)

        if predictor == "nps":
            return [
                max(
                    0.0,
                    min(1.0, (float(v) + 100.0) / 200.0),
                )
                for v in values
            ]

        return [
            max(
                0.0,
                min(1.0, float(v) / scale),
            )
            for v in values
        ]

    def analyze(
        self,
        predictor: str,
        observations: list[float],
        baseline: float,
        uncertainty: float | None = None,
        samples: int = 10000,
    ) -> PredictorProbabilisticResult:

        values = [float(v) for v in observations]

        metric_uncertainty = self._derive_uncertainty(
            predictor,
            values,
            uncertainty,
        )

        normalized = self._normalize(
            predictor,
            values,
        )

        # Bayesian and Monte Carlo operate in the same normalized probability
        # domain. The previous implementation normalized historical values
        # but passed the raw prediction and raw metric uncertainty to the
        # Monte Carlo engine, producing an unbounded Normal distribution
        # outside [0, 1]. Normalize both exactly once and bound the SAME
        # simulation draw to the probability domain.
        normalized_baseline = self._normalize(
            predictor,
            [float(baseline)],
        )[0]

        scale = 200.0 if predictor == "nps" else self._metric_scale(predictor)
        normalized_uncertainty = metric_uncertainty / scale

        universal_bayesian = self.core.from_bayesian(
            observations=normalized,
        )

        universal_monte_carlo = self.core.from_monte_carlo(
            baseline=float(normalized_baseline),
            uncertainty=float(normalized_uncertainty),
            bounds=(0.0, 1.0),
            samples=samples,
            metadata={
                "predictor": predictor,
                "probability_domain": True,
                "source_scale": scale,
            },
        )

        bayesian_info = universal_bayesian.bayesian

        monte_carlo_info = universal_monte_carlo.monte_carlo

        probability = (
            universal_bayesian.probability_of_target
            if universal_bayesian.probability_of_target is not None
            else 0.0
        )

        confidence = (
            universal_bayesian.confidence
            if universal_bayesian.confidence is not None
            else 0.0
        )

        expected = (
            universal_monte_carlo.expected_value
            if universal_monte_carlo.expected_value is not None
            else float(baseline)
        )

        downside = (
            universal_monte_carlo.likely_range_lower
            if universal_monte_carlo.likely_range_lower is not None
            else expected
        )

        upside = (
            universal_monte_carlo.likely_range_upper
            if universal_monte_carlo.likely_range_upper is not None
            else expected
        )

        simulations = (
            monte_carlo_info.num_simulations
            if monte_carlo_info is not None
            and monte_carlo_info.num_simulations is not None
            else samples
        )

        legacy_monte_carlo = SimpleNamespace(
            mean=float(expected),
            p05=float(downside),
            p50=float(
                universal_monte_carlo.most_likely
                if universal_monte_carlo.most_likely is not None
                else expected
            ),
            p95=float(upside),
            probability_positive=(
                float(universal_monte_carlo.probability_of_target)
                if universal_monte_carlo.probability_of_target is not None
                else 0.0
            ),
            samples=int(simulations),
            uncertainty=float(normalized_uncertainty),
            metadata=(
                monte_carlo_info.metadata
                if monte_carlo_info is not None
                else {}
            ),
        )

        legacy_bayesian = SimpleNamespace(
            probability=float(probability),
            confidence=float(confidence),
            posterior_mean=(
                float(bayesian_info.posterior_mean)
                if bayesian_info is not None
                and bayesian_info.posterior_mean is not None
                else float(probability)
            ),
            posterior_std=(
                float(bayesian_info.posterior_std)
                if bayesian_info is not None
                and bayesian_info.posterior_std is not None
                else 0.0
            ),
            samples=len(values),
        )

        return PredictorProbabilisticResult(
            predictor=predictor,
            bayesian=legacy_bayesian,
            monte_carlo=legacy_monte_carlo,
            probability=float(probability),
            confidence=float(confidence),
            expected=float(expected),
            downside=float(downside),
            upside=float(upside),
            simulations=int(simulations),
            uncertainty=float(metric_uncertainty),
            historical_samples=len(values),
            result=universal_monte_carlo,
        )

    def analyze_prediction(
        self,
        predictor: str,
        prediction: float,
        historical_values: list[float] | None = None,
        uncertainty: float | None = None,
        samples: int = 10000,
    ) -> PredictorProbabilisticResult:

        return self.analyze(
            predictor=predictor,
            observations=historical_values or [],
            baseline=float(prediction),
            uncertainty=uncertainty,
            samples=samples,
        )


_default_adapter = UniversalProbabilisticAdapter()


def analyze(
    predictor: str,
    observations: list[float],
    baseline: float,
    uncertainty: float | None = None,
    samples: int = 10000,
) -> PredictorProbabilisticResult:
    return _default_adapter.analyze(
        predictor=predictor,
        observations=observations,
        baseline=baseline,
        uncertainty=uncertainty,
        samples=samples,
    )


def analyze_prediction(
    predictor: str,
    prediction: float,
    historical_values: list[float] | None = None,
    uncertainty: float | None = None,
    samples: int = 10000,
) -> PredictorProbabilisticResult:
    return _default_adapter.analyze_prediction(
        predictor=predictor,
        prediction=prediction,
        historical_values=historical_values,
        uncertainty=uncertainty,
        samples=samples,
    )
