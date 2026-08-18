from __future__ import annotations

from typing import Any

from core.probabilistic import UniversalPredictionEnvelope, wrap_nps_prediction


def adapt_nps_prediction(
    prediction: float,
    *,
    score_distribution: dict[str, float] | list[float],
    total_surveys: int,
    observed_score_counts: list[int] | None = None,
    simulations: int = 10000,
    seed: int = 0,
    prior_strength: float = 20.0,
    metadata: dict[str, Any] | None = None,
) -> UniversalPredictionEnvelope:
    """Add probabilistic information to NPS using the canonical 0..10 path.

    Scalar NPS uncertainty is intentionally unsupported.
    """
    merged_metadata = {
        "predictor": "nps",
        "metric": "nps",
        "distribution_authoritative": True,
        **(metadata or {}),
    }

    return wrap_nps_prediction(
        float(prediction),
        score_distribution=score_distribution,
        total_surveys=int(total_surveys),
        observed_score_counts=observed_score_counts,
        simulations=int(simulations),
        seed=int(seed),
        prior_strength=float(prior_strength),
        metadata=merged_metadata,
    )


def adapt_nps_result(
    result: Any,
    *,
    score_distribution: dict[str, float] | list[float] | None = None,
    total_surveys: int | None = None,
    observed_score_counts: list[int] | None = None,
    simulations: int = 10000,
    seed: int = 0,
    prior_strength: float = 20.0,
) -> UniversalPredictionEnvelope:
    """Adapt an existing NPS result without ever creating scalar NPS uncertainty."""
    if isinstance(result, tuple):
        if len(result) < 2:
            raise ValueError("NPS tuple result must contain an NPS value")
        prediction = float(result[1])
        if score_distribution is None and len(result) >= 3:
            score_distribution = result[2]

    elif isinstance(result, dict):
        if "nps" not in result:
            raise ValueError("NPS result dictionary must contain 'nps'")
        prediction = float(result["nps"])
        if score_distribution is None:
            score_distribution = (
                result.get("bayesian_score_distribution")
                or result.get("score_distribution")
            )
        if observed_score_counts is None and result.get("score_counts"):
            counts = result["score_counts"]
            observed_score_counts = [
                int(counts.get(f"score_{i}", 0)) if isinstance(counts, dict) else int(counts[i])
                for i in range(11)
            ]
        if total_surveys is None:
            total_surveys = result.get("total_surveys")
            if total_surveys is None and observed_score_counts is not None:
                total_surveys = sum(observed_score_counts)

    else:
        prediction = float(result)

    if score_distribution is None or total_surveys is None:
        raise ValueError(
            "NPS probabilistic adaptation requires score_0..score_10 distribution "
            "and total_surveys; scalar NPS uncertainty is prohibited."
        )

    return adapt_nps_prediction(
        prediction,
        score_distribution=score_distribution,
        total_surveys=int(total_surveys),
        observed_score_counts=observed_score_counts,
        simulations=simulations,
        seed=seed,
        prior_strength=prior_strength,
    )
