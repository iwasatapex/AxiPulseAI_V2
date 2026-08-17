
"""
AxiPulseAI — Bayesian 0–10 NPS survey distribution.

Bayesian inference is performed over the categorical survey score
distribution 0..10.

It does NOT operate on the final NPS scalar.

Score buckets:
    0..6  = Detractors
    7..8  = Passives
    9..10 = Promoters
"""

from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np


N_SCORES = 11


def normalize_distribution(values):
    """Return a safe normalized 11-score probability vector."""

    x = np.asarray(values, dtype=float).reshape(-1)

    if x.size != N_SCORES:
        raise ValueError(
            f"Expected 11 score values, received {x.size}"
        )

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    x = np.clip(x, 0.0, None)

    total = float(x.sum())

    if total <= 0.0:
        x[:] = 1.0
        total = float(x.sum())

    return x / total


def observed_score_counts(row):
    """
    Extract real observed score counts.

    Only score_0 ... score_10 are accepted.

    Predicted values are never treated as observations.
    """

    if row is None:
        return None

    keys = [f"score_{i}" for i in range(N_SCORES)]

    if not all(key in row for key in keys):
        return None

    try:
        values = np.asarray(
            [float(row[key]) for key in keys],
            dtype=float,
        )
    except Exception:
        return None

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if np.any(values < 0):
        return None

    if float(values.sum()) <= 0:
        return None

    return values


def bayesian_update(
    predicted_distribution,
    observed_counts=None,
    prior_strength=20.0,
):
    """
    Dirichlet Bayesian update.

    Prior:
        ML-predicted score 0..10 distribution.

    Evidence:
        observed survey score counts.

    Posterior:
        updated probability for every score 0..10.

    If there is no observed evidence, the posterior equals
    the ML distribution.
    """

    if prior_strength <= 0:
        raise ValueError(
            "prior_strength must be greater than zero"
        )

    prior = normalize_distribution(
        predicted_distribution
    )

    if observed_counts is None:
        observed = np.zeros(N_SCORES, dtype=float)
    else:
        observed = np.asarray(
            observed_counts,
            dtype=float,
        ).reshape(-1)

        if observed.size != N_SCORES:
            raise ValueError(
                "Observed counts must contain scores 0..10"
            )

        observed = np.nan_to_num(
            observed,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        observed = np.clip(
            observed,
            0.0,
            None,
        )

    prior_alpha = prior * float(prior_strength)

    posterior_alpha = (
        prior_alpha + observed
    )

    posterior = (
        posterior_alpha
        / posterior_alpha.sum()
    )

    return {
        "prior": prior,
        "observed": observed,
        "posterior": posterior,
        "prior_strength": float(prior_strength),
        "observed_count": int(
            round(float(observed.sum()))
        ),
    }


def posterior_counts(
    posterior_distribution,
    total_surveys,
):
    """
    Convert posterior probabilities into integer survey counts.

    Largest remainder method guarantees the counts sum exactly
    to total_surveys.
    """

    total_surveys = int(total_surveys)

    if total_surveys <= 0:
        return np.zeros(
            N_SCORES,
            dtype=int,
        )

    p = normalize_distribution(
        posterior_distribution
    )

    raw = p * total_surveys

    counts = np.floor(raw).astype(int)

    remaining = (
        total_surveys
        - int(counts.sum())
    )

    if remaining > 0:
        fractional = raw - counts

        order = np.argsort(
            -fractional
        )

        for idx in order[:remaining]:
            counts[idx] += 1

    return counts


def nps_from_score_counts(
    score_counts,
):
    """
    Calculate NPS only after the 0–10 distribution exists.
    """

    counts = np.asarray(
        score_counts,
        dtype=int,
    ).reshape(-1)

    if counts.size != N_SCORES:
        raise ValueError(
            "Expected 11 score counts"
        )

    detractors = int(
        counts[:7].sum()
    )

    passives = int(
        counts[7:9].sum()
    )

    promoters = int(
        counts[9:11].sum()
    )

    total = (
        detractors
        + passives
        + promoters
    )

    nps = (
        ((promoters - detractors) / total)
        * 100.0
        if total > 0
        else 0.0
    )

    return {
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "nps": float(nps),
    }
