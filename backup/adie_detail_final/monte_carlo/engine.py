from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    mean: float
    p05: float
    p50: float
    p95: float
    probability_positive: float
    samples: int
    uncertainty: float
    expected_value: float | None = None
    variance: float | None = None
    tail_loss_probability: float | None = None
    # Additive sample summary retained from the SAME single simulation
    # (no second execution). ``bin_summary`` is a coarse histogram over the
    # realized sample range; success/failure counts partition the same draw.
    success_count: int = 0
    failure_count: int = 0
    distribution: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_inputs(
    baseline: float,
    uncertainty: float,
    samples: int,
    seed: int,
) -> None:
    if not np.isfinite(baseline):
        raise ValueError("baseline must be finite")

    if not np.isfinite(uncertainty):
        raise ValueError("uncertainty must be finite")

    if uncertainty < 0.0:
        raise ValueError(
            "uncertainty must be non-negative"
        )

    if samples <= 0:
        raise ValueError("samples must be positive")

    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")


def _bin_summary(values: np.ndarray) -> list[dict[str, Any]]:
    """Coarse histogram retained from the existing single simulation.

    Derived purely from ``values`` already in memory — no new Monte Carlo
    execution. Returns ~10 equal-width bins over the realized [min, max] range.
    Empty/all-identical samples yield an empty summary (no fabricated bins).
    """
    if values.size == 0:
        return []
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax - vmin <= 0.0:
        return []
    counts, edges = np.histogram(values, bins=10, range=(vmin, vmax))
    total = int(values.size)
    summary: list[dict[str, Any]] = []
    for idx in range(len(counts)):
        start = float(edges[idx])
        end = float(edges[idx + 1])
        count = int(counts[idx])
        summary.append({
            "bin_start": round(start, 6),
            "bin_end": round(end, 6),
            "count": count,
            "probability": round(count / total, 6) if total else 0.0,
        })
    return summary


def _summarize(
    values: np.ndarray,
    uncertainty: float,
    seed: int,
    distribution: str,
    metadata: dict[str, Any] | None = None,
) -> MonteCarloResult:
    positive = values > 0
    success = int(np.count_nonzero(positive))
    failure = int(values.size - success)
    return MonteCarloResult(
        mean=float(np.mean(values)),
        p05=float(np.percentile(values, 5)),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        probability_positive=float(np.mean(positive)),
        samples=int(values.size),
        uncertainty=float(uncertainty),
        expected_value=float(np.mean(values)),
        variance=float(np.var(values)),
        tail_loss_probability=float(np.mean(values < 0)),
        success_count=success,
        failure_count=failure,
        distribution=_bin_summary(values),
        metadata={
            "distribution": distribution,
            "seed": int(seed),
            **(metadata or {}),
        },
    )


def simulate(
    baseline: float,
    uncertainty: float = 0.05,
    samples: int = 10000,
    seed: int = 0,
    distribution: str = "normal",
) -> MonteCarloResult:
    """
    Universal Monte Carlo simulation.

    Supported distributions:
      - normal
      - uniform

    The API remains backward compatible with the original normal simulator.
    """
    _validate_inputs(
        baseline,
        uncertainty,
        samples,
        seed,
    )

    if distribution not in {"normal", "uniform"}:
        raise ValueError(
            "distribution must be 'normal' or 'uniform'"
        )

    rng = np.random.default_rng(seed)

    if distribution == "normal":
        values = rng.normal(
            loc=float(baseline),
            scale=float(uncertainty),
            size=int(samples),
        )
    else:
        half_width = float(uncertainty)
        values = rng.uniform(
            low=float(baseline) - half_width,
            high=float(baseline) + half_width,
            size=int(samples),
        )

    return _summarize(
        values=values,
        uncertainty=uncertainty,
        seed=seed,
        distribution=distribution,
    )


class MonteCarloEngine:
    """
    Universal domain-neutral Monte Carlo engine.
    """

    def simulate(
        self,
        baseline: float,
        uncertainty: float = 0.05,
        samples: int = 10000,
        seed: int = 0,
        distribution: str = "normal",
    ) -> MonteCarloResult:
        return simulate(
            baseline=baseline,
            uncertainty=uncertainty,
            samples=samples,
            seed=seed,
            distribution=distribution,
        )

    def run(
        self,
        sampler: Callable[[np.random.Generator, int], Any],
        samples: int = 10000,
        seed: int = 0,
        uncertainty: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> MonteCarloResult:
        """
        Run a caller-supplied distribution sampler.

        This keeps the core engine extensible without embedding
        domain-specific distributions.
        """
        if samples <= 0:
            raise ValueError("samples must be positive")

        if not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")

        if not np.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError(
                "uncertainty must be finite and non-negative"
            )

        rng = np.random.default_rng(seed)
        values = np.asarray(
            sampler(rng, int(samples)),
            dtype=float,
        )

        if values.ndim != 1:
            raise ValueError("sampler must return one-dimensional values")

        if values.size != samples:
            raise ValueError(
                "sampler returned an unexpected number of samples"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "sampler returned non-finite values"
            )

        return _summarize(
            values=values,
            uncertainty=uncertainty,
            seed=seed,
            distribution="custom",
            metadata=metadata,
        )
