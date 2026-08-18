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
    """Result of a single Bayesian 0..10 Dirichlet update."""

    posterior_mean: float = 0.0
    posterior: np.ndarray | None = None
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

    result = BayesianResult(
        posterior_mean=mean_score,
        posterior=posterior,
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
) -> dict[str, Any]:
    """Monte Carlo simulation on the 0..10 score distribution."""
    # Extract posterior from BayesianResult if needed
    if isinstance(posterior_distribution, BayesianResult):
        posterior = _extract_posterior(posterior_distribution)
    else:
        posterior = np.asarray(posterior_distribution, dtype=float).reshape(-1)

    if len(posterior) != N_SCORES:
        raise ValueError(f"Expected {N_SCORES} posterior probabilities, got {len(posterior)}")

    # Normalize
    posterior = np.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)
    posterior = np.maximum(posterior, 0.0)
    total = float(posterior.sum())
    if total <= 0.0:
        posterior = np.full(N_SCORES, 1.0 / N_SCORES, dtype=float)
    else:
        posterior = posterior / total

    # Monte Carlo draws
    rng = np.random.default_rng(random_state)

    simulation_nps = np.empty(simulations, dtype=float)
    all_scores = np.empty((simulations, total_surveys), dtype=int)

    for sim_idx in range(simulations):
        scores = rng.choice(np.arange(N_SCORES), size=total_surveys, replace=True, p=posterior)
        all_scores[sim_idx] = scores
        detractors = int(np.sum((scores >= 0) & (scores <= 6)))
        passives = int(np.sum((scores >= 7) & (scores <= 8)))
        promoters = int(np.sum((scores >= 9) & (scores <= 10)))
        nps = ((promoters - detractors) / total_surveys) * 100.0
        simulation_nps[sim_idx] = nps

    # Compute mean_score_counts across all draws
    total_draws = simulations * total_surveys
    score_counts = np.zeros(N_SCORES, dtype=float)
    for s in range(N_SCORES):
        score_counts[s] = float(np.sum(all_scores == s))
    mean_score_counts = score_counts / total_draws * total_surveys  # per-survey frequency

    p05 = float(np.percentile(simulation_nps, 5))
    p50 = float(np.percentile(simulation_nps, 50))
    p95 = float(np.percentile(simulation_nps, 95))

    return {
        "simulation_nps": simulation_nps,
        "mean_score_counts": mean_score_counts,
        "p05": p05,
        "p50": p50,
        "p95": p95,
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
) -> dict:
    """Attach canonical categorical Bayesian/Monte Carlo evidence."""
    import copy

    result = copy.deepcopy(result)

    raw_distribution = result.get("bayesian_score_distribution")
    if not isinstance(raw_distribution, dict):
        return result

    # Get the 11-score distribution
    ml_distribution = np.array(
        [float(raw_distribution.get(f"score_{i}", 0.0)) for i in range(N_SCORES)],
        dtype=float,
    )

    # Normalize
    ml_distribution = np.nan_to_num(ml_distribution, nan=0.0, posinf=0.0, neginf=0.0)
    ml_distribution = np.maximum(ml_distribution, 0.0)
    total = float(ml_distribution.sum())
    if total <= 0.0:
        ml_distribution = np.full(N_SCORES, 1.0 / N_SCORES, dtype=float)
    else:
        ml_distribution = ml_distribution / total

    # Bayesian update
    if observed_counts is not None:
        obs = np.asarray(observed_counts, dtype=float).reshape(-1)
        if obs.size != N_SCORES:
            raise ValueError(f"Expected {N_SCORES} observed counts, got {obs.size}")
        alpha = ml_distribution * float(max(1.0, 10.0)) + obs
        posterior = alpha / float(alpha.sum())
    else:
        posterior = ml_distribution.copy()

    # Monte Carlo - pass the posterior as list
    mc_result = from_monte_carlo(
        posterior.tolist(),
        total_surveys=total_surveys,
        simulations=simulations,
        random_state=seed,
    )

    # Update result dict
    result["bayesian_score_distribution"] = {
        f"score_{i}": float(posterior[i]) for i in range(N_SCORES)
    }
    result["monte_carlo_score_distribution"] = {
        f"score_{i}": float(mc_result["mean_score_counts"][i]) for i in range(N_SCORES)
    }

    # Preserve the model/business point estimate and bucket counts already
    # produced by postprocess_predictions. Monte Carlo is uncertainty evidence,
    # not a replacement for the deterministic NPS point estimate.
    result["monte_carlo_nps"] = mc_result["simulation_nps"].tolist()
    result["monte_carlo_nps_p05"] = mc_result["p05"]
    result["monte_carlo_nps_p50"] = mc_result["p50"]
    result["monte_carlo_nps_p95"] = mc_result["p95"]

    return result
