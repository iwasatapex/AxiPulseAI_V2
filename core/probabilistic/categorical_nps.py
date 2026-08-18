"""
Canonical categorical (0..10 NPS) Bayesian and Monte Carlo inference.

This module replaces the private ``core.nps_predictor.inference._axi_*`` implementations
and is the sole authoritative execution path for NPS 0..10 probabilistic analysis.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np


N_SCORES = 11  # scores 0..10


class BayesianResult:
    """Result of a Bayesian Dirichlet model over survey scores 0..10."""

    posterior_mean: float = 0.0
    posterior: np.ndarray | None = None
    posterior_alpha: np.ndarray | None = None
    prior_mean: float = 0.0
    prior_strength: float = 0.0
    credible_interval_lower: float | None = None
    credible_interval_upper: float | None = None
    credible_level: float = 0.95
    prior_type: str = "dirichlet"
    metadata: dict = None

    def __init__(self, **kwargs):
        # Set all provided fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Set defaults for any unset fields
        if self.posterior is None:
            self.posterior = np.zeros(N_SCORES, dtype=float)
        if self.posterior_alpha is None:
            self.posterior_alpha = np.asarray(self.posterior, dtype=float).copy()
        if self.metadata is None:
            self.metadata = {}

    def __repr__(self) -> str:
        if self.posterior is not None:
            mean_score = float(np.sum(np.arange(N_SCORES) * self.posterior))
        else:
            mean_score = self.posterior_mean
        return (
            f"BayesianResult(posterior_mean={mean_score:.4f}, "
            f"p05={self.credible_interval_lower}, p95={self.credible_interval_upper})")


def _extract_posterior(result: BayesianResult | list[float] | np.ndarray) -> np.ndarray:
    """Extract the posterior array from a BayesianResult or return as-is."""
    if isinstance(result, BayesianResult):
        if result.posterior is not None:
            return result.posterior
        return np.zeros(N_SCORES, dtype=float)
    arr = np.asarray(result, dtype=float).reshape(-1)
    if len(arr) != N_SCORES:
        raise ValueError(f"Expected {N_SCORES} values, got {len(arr)}")
    return arr


def _validate_prior_strength(prior_strength: float) -> None:
    if not np.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength must be greater than 0")


def _validate_credible_level(credible_level: float) -> None:
    if not np.isfinite(credible_level) or not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be between 0 and 1")


def from_prior_only(
    prior_distribution: list[float] | np.ndarray,
    prior_strength: float = 10.0,
    credible_level: float = 0.95,
    metadata: dict | None = None,
) -> BayesianResult:
    """Bayesian update when NO observed counts are supplied."""
    _validate_prior_strength(prior_strength)
    _validate_credible_level(credible_level)

    prior = np.asarray(prior_distribution, dtype=float).reshape(-1)
    if len(prior) != N_SCORES:
        raise ValueError(f"Expected {N_SCORES} prior probabilities, got {len(prior)}")

    # Normalize (same as _axi_normalize_score_distribution)
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    prior = np.maximum(prior, 0.0)
    total = float(prior.sum())
    if total <= 0.0:
        prior = np.full(N_SCORES, 1.0 / N_SCORES, dtype=float)
    else:
        prior = prior / total

    # With no observed evidence, posterior = prior
    posterior = prior.copy()

    # Compute mean score
    mean_score = float(np.sum(np.arange(N_SCORES) * posterior))

    # Approximate credible interval from posterior std
    eff_n = float(prior_strength) + float(np.sum(prior > 0))  # effective count
    posterior_std = sqrt(float(np.sum((np.arange(N_SCORES) - mean_score) ** 2 * posterior)) / (eff_n + 1)) if eff_n > 0 else 0.0

    ci_lower = float(np.clip(mean_score - 1.96 * posterior_std, 0, 10))
    ci_upper = float(np.clip(mean_score + 1.96 * posterior_std, 0, 10))

    if metadata is None:
        metadata = {}
    metadata["prior_strength"] = float(prior_strength)

    posterior_alpha = prior * float(prior_strength)

    result = BayesianResult(
        posterior_mean=mean_score,
        posterior=posterior,
        posterior_alpha=posterior_alpha,
        prior_mean=float(np.mean(prior)),
        prior_strength=float(prior_strength),
        credible_interval_lower=ci_lower,
        credible_interval_upper=ci_upper,
        credible_level=credible_level,
        metadata=metadata,
    )

    return result


def from_observed_counts(
    prior_distribution: list[float] | np.ndarray,
    observed_counts: list[int] | np.ndarray,
    prior_strength: float = 10.0,
    credible_level: float = 0.95,
    metadata: dict | None = None,
) -> BayesianResult:
    """Bayesian update WITH observed score counts."""
    _validate_prior_strength(prior_strength)
    _validate_credible_level(credible_level)

    prior = np.asarray(prior_distribution, dtype=float).reshape(-1)
    if len(prior) != N_SCORES:
        raise ValueError(f"Expected {N_SCORES} prior probabilities, got {len(prior)}")

    # Normalize prior
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    prior = np.maximum(prior, 0.0)
    total = float(prior.sum())
    if total <= 0.0:
        prior = np.full(N_SCORES, 1.0 / N_SCORES, dtype=float)
    else:
        prior = prior / total

    # Original implementation: alpha = prior * prior_strength + counts
    obs = np.asarray(observed_counts, dtype=float).reshape(-1)
    if obs.size != N_SCORES:
        raise ValueError(f"Expected {N_SCORES} observed counts, got {obs.size}")
    if np.any(~np.isfinite(obs)) or np.any(obs < 0):
        raise ValueError("Observed score counts must be finite and non-negative")

    alpha = prior * float(max(1.0, prior_strength)) + obs
    posterior = alpha / float(alpha.sum())

    # Compute mean score and credible interval
    mean_score = float(np.sum(np.arange(N_SCORES) * posterior))
    eff_n = float(alpha.sum())
    posterior_std = sqrt(float(np.sum((np.arange(N_SCORES) - mean_score) ** 2 * posterior)) / (eff_n + 1)) if eff_n > 0 else 0.0

    ci_lower = float(np.clip(mean_score - 1.96 * posterior_std, 0, 10))
    ci_upper = float(np.clip(mean_score + 1.96 * posterior_std, 0, 10))

    if metadata is None:
        metadata = {}
    metadata["prior_strength"] = float(prior_strength)
    metadata["observation_count"] = int(np.nansum(obs)) if np.all(np.isfinite(obs)) else 0
    metadata["success_mass"] = float(obs.sum()) if np.all(np.isfinite(obs)) else 0.0
    metadata["failure_mass"] = float(N_SCORES) - float(np.nansum(obs)) if np.all(np.isfinite(obs)) else 0.0

    result = BayesianResult(
        posterior_mean=mean_score,
        posterior=posterior,
        posterior_alpha=alpha,
        prior_mean=float(np.mean(prior)),
        prior_strength=float(prior_strength),
        credible_interval_lower=ci_lower,
        credible_interval_upper=ci_upper,
        credible_level=credible_level,
        metadata=metadata,
    )

    return result


def from_monte_carlo(
    posterior_distribution: list[float] | np.ndarray | BayesianResult,
    total_surveys: int,
    simulations: int = 1000,
    random_state: int = 42,
    dirichlet_strength: float | None = None,
) -> dict[str, Any]:
    """Propagate uncertainty through the 0..10 survey-score distribution.

    Monte Carlo operates on survey scores, never on a scalar NPS. If a
    ``BayesianResult`` is supplied, each simulation first draws a score
    probability vector from the Dirichlet posterior, then draws survey counts
    from that vector. This propagates both Bayesian parameter uncertainty and
    finite survey-sample uncertainty into NPS.
    """
    total_surveys = int(total_surveys)
    simulations = int(simulations)
    if total_surveys <= 0:
        raise ValueError("total_surveys must be greater than zero")
    if simulations <= 0:
        raise ValueError("simulations must be greater than zero")

    posterior_alpha = None
    if isinstance(posterior_distribution, BayesianResult):
        posterior = _extract_posterior(posterior_distribution)
        posterior_alpha = np.asarray(
            posterior_distribution.posterior_alpha, dtype=float
        ).reshape(-1)
        if posterior_alpha.size != N_SCORES or np.any(posterior_alpha <= 0):
            posterior_alpha = None
    else:
        posterior = np.asarray(posterior_distribution, dtype=float).reshape(-1)

    if posterior.size != N_SCORES:
        raise ValueError(
            f"Expected {N_SCORES} posterior probabilities, got {posterior.size}"
        )

    posterior = np.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)
    posterior = np.maximum(posterior, 0.0)
    total = float(posterior.sum())
    if total <= 0.0:
        posterior = np.full(N_SCORES, 1.0 / N_SCORES, dtype=float)
    else:
        posterior = posterior / total

    if posterior_alpha is None and dirichlet_strength is not None:
        if not np.isfinite(dirichlet_strength) or dirichlet_strength <= 0:
            raise ValueError("dirichlet_strength must be greater than zero")
        posterior_alpha = posterior * float(dirichlet_strength)

    rng = np.random.default_rng(random_state)
    simulation_nps = np.empty(simulations, dtype=float)
    total_score_counts = np.zeros(N_SCORES, dtype=float)

    for sim_idx in range(simulations):
        # Bayesian parameter uncertainty: draw the 0..10 probability vector.
        if posterior_alpha is not None:
            probabilities = rng.dirichlet(posterior_alpha)
        else:
            probabilities = posterior

        # Sampling uncertainty: draw the actual survey counts.
        counts = rng.multinomial(total_surveys, probabilities)
        total_score_counts += counts

        detractors = int(counts[:7].sum())
        promoters = int(counts[9:11].sum())
        simulation_nps[sim_idx] = (
            (promoters - detractors) / total_surveys
        ) * 100.0

    mean_score_counts = (
        total_score_counts / float(simulations)
    )

    return {
        "simulation_nps": simulation_nps,
        "mean_score_counts": mean_score_counts,
        "p05": float(np.percentile(simulation_nps, 5)),
        "p50": float(np.percentile(simulation_nps, 50)),
        "p95": float(np.percentile(simulation_nps, 95)),
        "simulation_count": simulations,
        "total_surveys": total_surveys,
        "distribution_domain": "survey_scores_0_10",
        "bayesian_parameter_sampling": posterior_alpha is not None,
    }


def nps_from_score_counts(
    score_counts: list[int] | np.ndarray,
) -> dict[str, int | float]:
    """Calculate NPS from 0..10 score counts."""
    counts = np.asarray(score_counts, dtype=int).reshape(-1)
    if counts.size != N_SCORES:
        raise ValueError(f"Expected {N_SCORES} score counts, got {counts.size}")

    detractors = int(counts[:7].sum())
    passives = int(counts[7:9].sum())
    promoters = int(counts[9:11].sum())
    total = detractors + passives + promoters

    nps = ((promoters - detractors) / total) * 100.0 if total > 0 else 0.0

    return {
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "nps": float(nps),
        "total_surveys": total,
    }


def attach_probabilistic_analysis(
    result: dict,
    total_surveys: int,
    observed_counts: list[int] | None = None,
    simulations: int = 1000,
    seed: int = 42,
    prior_strength: float = 20.0,
) -> dict:
    """Attach canonical Bayesian + Monte Carlo analysis on scores 0..10.

    The ML model supplies a distribution over the eleven survey scores.
    Bayesian inference updates that distribution using real observed score
    counts when available; when forecasting future outcomes, the ML
    distribution is the Bayesian prior because future survey counts do not yet
    exist. Monte Carlo then propagates the Bayesian score uncertainty and
    survey-sampling uncertainty into a distribution of NPS values.
    """
    import copy

    result = copy.deepcopy(result)
    raw_distribution = result.get("bayesian_score_distribution")
    if not isinstance(raw_distribution, dict):
        raise ValueError("result must contain bayesian_score_distribution")

    ml_distribution = np.array(
        [float(raw_distribution.get(f"score_{i}", 0.0)) for i in range(N_SCORES)],
        dtype=float,
    )

    if observed_counts is None:
        bayesian = from_prior_only(
            ml_distribution,
            prior_strength=prior_strength,
            metadata={"evidence": "model_distribution_only"},
        )
    else:
        bayesian = from_observed_counts(
            ml_distribution,
            observed_counts,
            prior_strength=prior_strength,
            metadata={"evidence": "observed_score_counts"},
        )

    mc_result = from_monte_carlo(
        bayesian,
        total_surveys=total_surveys,
        simulations=simulations,
        random_state=seed,
    )

    posterior = np.asarray(bayesian.posterior, dtype=float)
    posterior /= posterior.sum()

    result["bayesian_score_distribution"] = {
        f"score_{i}": float(posterior[i]) for i in range(N_SCORES)
    }
    result["bayesian_posterior_alpha"] = {
        f"score_{i}": float(bayesian.posterior_alpha[i])
        for i in range(N_SCORES)
    }
    result["bayesian_prior_strength"] = float(prior_strength)
    result["bayesian_evidence"] = bayesian.metadata.get("evidence")

    result["monte_carlo_score_distribution"] = {
        f"score_{i}": float(mc_result["mean_score_counts"][i])
        for i in range(N_SCORES)
    }
    result["monte_carlo_nps"] = mc_result["simulation_nps"].tolist()
    result["monte_carlo_nps_p05"] = mc_result["p05"]
    result["monte_carlo_nps_p50"] = mc_result["p50"]
    result["monte_carlo_nps_p95"] = mc_result["p95"]
    result["monte_carlo_simulations"] = simulations
    result["probabilistic_domain"] = "survey_scores_0_10"
    result["monte_carlo_bayesian_parameter_sampling"] = bool(
        mc_result["bayesian_parameter_sampling"]
    )

    # NPS remains derived from the score distribution/counts, never used as a
    # probability-model input. For forecasts, the deterministic point estimate
    # remains the NPS calculated from the Bayesian/posterior expected counts.
    posterior_expected_counts = posterior * int(total_surveys)
    point_counts = np.floor(posterior_expected_counts).astype(int)
    remainder = int(total_surveys) - int(point_counts.sum())
    if remainder > 0:
        fractional = posterior_expected_counts - point_counts
        for idx in np.argsort(-fractional)[:remainder]:
            point_counts[idx] += 1

    point = nps_from_score_counts(point_counts)
    result["probabilistic_score_counts"] = {
        f"score_{i}": int(point_counts[i]) for i in range(N_SCORES)
    }
    result["probabilistic_promoters"] = point["promoters"]
    result["probabilistic_passives"] = point["passives"]
    result["probabilistic_detractors"] = point["detractors"]
    result["probabilistic_nps"] = point["nps"]
    result["prediction_interval"] = {
        "low": mc_result["p05"],
        "high": mc_result["p95"],
    }

    return result

