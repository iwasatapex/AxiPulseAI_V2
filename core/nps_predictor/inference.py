

# AXIPULSEAI_BAYESIAN_0_10_PRODUCTION_LAYER
#
# IMPORTANT:
# Bayesian and Monte Carlo operate on the 11 individual survey scores.
# They do NOT operate on the already-aggregated NPS number.
#
# Score space:
#   0..6  = detractor
#   7..8  = passive
#   9..10 = promoter
#
# NPS is calculated only after the individual-score distribution exists.

def _axi_normalize_score_distribution(values):
    import numpy as np

    arr = np.asarray(values, dtype=float).reshape(-1)

    if len(arr) != 11:
        raise ValueError(
            f"Expected 11 NPS score probabilities, got {len(arr)}"
        )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    arr = np.maximum(arr, 0.0)

    total = float(arr.sum())

    if total <= 0.0:
        arr = np.full(
            11,
            1.0 / 11.0,
            dtype=float,
        )
    else:
        arr /= total

    return arr


def _axi_bayesian_update_0_10(
    prior_distribution,
    observed_counts=None,
    prior_strength=10.0,
):
    """
    Bayesian update over the actual NPS score space 0..10.

    The posterior is a categorical/Dirichlet update:

        posterior_i ∝ prior_i * prior_strength + observed_i

    If no observed score counts exist, the normalized ML distribution
    remains the predictive distribution rather than inventing evidence.

    This function NEVER accepts or operates on an NPS scalar.
    """
    import numpy as np

    prior = _axi_normalize_score_distribution(
        prior_distribution
    )

    alpha = (
        prior * float(max(1.0, prior_strength))
    )

    if observed_counts is not None:
        counts = np.asarray(
            observed_counts,
            dtype=float,
        ).reshape(-1)

        if len(counts) != 11:
            raise ValueError(
                "Bayesian evidence must contain 11 score counts."
            )

        counts = np.nan_to_num(
            counts,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        counts = np.maximum(counts, 0.0)

        alpha = alpha + counts

    posterior = alpha / float(alpha.sum())

    return posterior


def _axi_monte_carlo_0_10(
    posterior_distribution,
    total_surveys,
    simulations=1000,
    random_state=42,
):
    """
    Monte Carlo operates directly on 0..10 survey scores.

    Returns:
        simulation_nps
        score_counts
        bucket_counts
    """
    import numpy as np

    posterior = _axi_normalize_score_distribution(
        posterior_distribution
    )

    total_surveys = int(max(1, total_surveys))
    simulations = int(max(1, simulations))

    rng = np.random.default_rng(
        random_state
    )

    simulation_nps = np.empty(
        simulations,
        dtype=float,
    )

    final_counts = np.zeros(
        11,
        dtype=int,
    )

    # One simulation = one possible population of survey scores.
    for simulation_index in range(simulations):

        scores = rng.choice(
            np.arange(11),
            size=total_surveys,
            replace=True,
            p=posterior,
        )

        counts = np.bincount(
            scores,
            minlength=11,
        )

        final_counts += counts

        detractors = int(
            counts[0:7].sum()
        )

        passives = int(
            counts[7:9].sum()
        )

        promoters = int(
            counts[9:11].sum()
        )

        simulation_nps[simulation_index] = (
            (
                promoters
                - detractors
            )
            / float(total_surveys)
            * 100.0
        )

    # Average simulated score distribution.
    mean_counts = (
        final_counts.astype(float)
        / float(simulations)
    )

    return {
        "simulation_nps": simulation_nps,
        "mean_score_counts": mean_counts,
        "p05": float(
            np.percentile(
                simulation_nps,
                5,
            )
        ),
        "p50": float(
            np.percentile(
                simulation_nps,
                50,
            )
        ),
        "p95": float(
            np.percentile(
                simulation_nps,
                95,
            )
        ),
    }


def _axi_attach_0_10_probabilistic_analysis(
    result,
    total_surveys,
    observed_counts=None,
    simulations=1000,
):
    """
    Attach the complete production probabilistic analysis to the
    existing NPS result dictionary.

    Existing ML prediction and business bucket logic remain intact.
    """
    import numpy as np

    if not isinstance(result, dict):
        return result

    raw_distribution = result.get(
        "bayesian_score_distribution"
    )

    if not isinstance(
        raw_distribution,
        dict,
    ):
        return result

    ml_distribution = np.array(
        [
            float(
                raw_distribution.get(
                    f"score_{i}",
                    0.0,
                )
            )
            for i in range(11)
        ],
        dtype=float,
    )

    ml_distribution = _axi_normalize_score_distribution(
        ml_distribution
    )

    # Use actual observed 0..10 counts when the caller supplies them.
    # Otherwise Bayesian predictive distribution remains based on
    # the model evidence rather than fabricating observations.
    posterior = _axi_bayesian_update_0_10(
        ml_distribution,
        observed_counts=observed_counts,
    )

    mc = _axi_monte_carlo_0_10(
        posterior,
        total_surveys=total_surveys,
        simulations=simulations,
    )

    result[
        "bayesian_score_distribution"
    ] = {
        f"score_{i}": float(
            posterior[i]
        )
        for i in range(11)
    }

    result[
        "monte_carlo_score_distribution"
    ] = {
        f"score_{i}": float(
            mc["mean_score_counts"][i]
        )
        for i in range(11)
    }

    result[
        "monte_carlo_nps"
    ] = {
        "simulations": simulations,
        "p05": mc["p05"],
        "p50": mc["p50"],
        "p95": mc["p95"],
    }

    result[
        "probabilistic_score_space"
    ] = "0..10"

    result[
        "bayesian_operates_on"
    ] = "individual_nps_scores_0_to_10"

    result[
        "monte_carlo_operates_on"
    ] = "individual_nps_scores_0_to_10"

    # Business counts remain derived from the individual-score model.
    result[
        "posterior_promoter_probability"
    ] = float(
        posterior[9:11].sum()
    )

    result[
        "posterior_passive_probability"
    ] = float(
        posterior[7:9].sum()
    )

    result[
        "posterior_detractor_probability"
    ] = float(
        posterior[0:7].sum()
    )

    return result


from tqdm import tqdm

"""
AxiPulseAI – Inference (11-score distribution)
"""
import numpy as np
import pandas as pd
import logging

from .feature_engineering import align_features
from .models import compute_ensemble_weights

logger = logging.getLogger(__name__)


def _row_float(row, key, default):
    """Return a numeric value from a raw row dict, or ``default`` if missing /
    non-convertible. Inference rows may carry numeric fields as strings (e.g.
    ``operational_health="75.5"``); this mirrors the feature-matrix coercion so
    downstream count math operates on floats, never on ``object``/``str``.
    """
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    return value

def fallback_predict(predictor, row=None):
    # default distribution: 70% promoters, 20% passives, 10% detractors (similar to 0-10)
    total_surveys = 10
    if row is not None:
        total_calls = _row_float(row, "total_calls_received", 100)
        release_rate = _row_float(row, "actual_release_rate", 60.0)
        released_calls = max(1, int(round(total_calls * release_rate / 100)))
        health = _row_float(row, "operational_health", 80.0)
        health_clamped = max(0, min(120, health))
        survey_rate = 0.08 + (health_clamped / 120) * 0.03
        survey_rate = np.clip(survey_rate, 0.08, 0.11)
        total_surveys = max(1, int(round(released_calls * survey_rate)))
    promoters = int(round(0.70 * total_surveys))
    passives = int(round(0.20 * total_surveys))
    detractors = total_surveys - promoters - passives
    nps = ((promoters - detractors) / total_surveys) * 100 if total_surveys > 0 else 0.0
    return {
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "nps": nps,
        "confidence": 50.0,
        "prediction_interval": {"low": nps - 10, "high": nps + 10},
        "top_drivers": []
    }




def postprocess_predictions(pred_pct, row, preds=None, strict=False):
    """
    Convert predicted percentages into realistic survey counts.

    Survey count is derived from released calls and the cutoff-known
    operational state. The final NPS distribution itself comes from the
    trained 0..10 score model; there is no hidden KPI-to-NPS forcing rule.
    """
    import numpy as np

    total_calls = _row_float(row, "total_calls_received", 100)
    release_rate = _row_float(row, "actual_release_rate", 60.0)
    released_calls = max(1, int(round(total_calls * release_rate / 100)))

    # ---- 1. Survey count: 10-17% based on health ----
    health = _row_float(row, "operational_health", 80.0)
    health_clamped = max(0, min(120, health))
    survey_rate = 0.08 + (health_clamped / 120) * 0.03  # 0.10..0.17
    survey_rate = np.clip(survey_rate, 0.08, 0.11)
    total_surveys = max(1, int(round(released_calls * survey_rate)))

    # ---- 2. ML distribution (11-score output) ----
    scores = np.asarray(pred_pct, dtype=float).flatten()

    if scores.size != 11:
        raise RuntimeError(f"Expected 11 outputs, got {scores.size}")

    scores = np.maximum(scores, 0)

    if scores.sum() == 0:
        scores[:] = 1.0

    scores /= scores.sum()

    # ---- 2A. Bayesian update on the 0-10 score distribution ----
    #
    # Bayesian reasoning belongs at the individual survey-score level.
    # The ML model supplies the observed 0..10 probability distribution.
    # A symmetric Dirichlet prior prevents any score from becoming
    # artificially impossible and produces a posterior distribution.
    #
    # The resulting posterior is used for the score distribution only.
    # NPS remains calculated later from the resulting survey counts.
    bayesian_prior_strength = 1.0
    bayesian_evidence_strength = max(
        1.0,
        float(total_surveys),
    )

    bayesian_prior = np.full(
        11,
        bayesian_prior_strength / 11.0,
        dtype=float,
    )

    bayesian_evidence = (
        scores
        * bayesian_evidence_strength
    )

    bayesian_posterior = (
        bayesian_prior
        + bayesian_evidence
    )

    bayesian_posterior /= (
        bayesian_posterior.sum()
    )

    # Preserve the universal 0..10 score distribution.
    scores = bayesian_posterior

    # Recalculate business buckets AFTER Bayesian updating.
    detractor_pct = scores[:7].sum()
    passive_pct = scores[7:9].sum()
    promoter_pct = scores[9:11].sum()


    ml_dist = np.array(
        [promoter_pct, passive_pct, detractor_pct],
        dtype=float,
    )

    ml_dist /= ml_dist.sum()

    # ---- 3. Production NPS buckets come directly from the learned
    # 0..10 survey-score distribution. No hard-coded KPI-to-NPS target is
    # blended into the model output.
    ml_dist = np.array(
        [promoter_pct, passive_pct, detractor_pct],
        dtype=float,
    )
    ml_dist /= ml_dist.sum()
    final_dist = ml_dist

    # ==========================================================
    # 5. Convert the 3 business buckets back onto the
    #    Bayesian 0–10 score distribution.
    #
    # Bayesian operates on:
    #
    #   score 0 ... score 10
    #
    # Existing business logic operates on:
    #
    #   final_dist[0] = promoter probability
    #   final_dist[1] = passive probability
    #   final_dist[2] = detractor probability
    #
    # Preserve BOTH:
    #
    #   - business bucket probabilities
    #   - individual Bayesian score probabilities
    #
    # Bucket mapping:
    #
    #   0–6  -> Detractors
    #   7–8  -> Passives
    #   9–10 -> Promoters
    # ==========================================================

    bayesian_score_dist = np.asarray(
        scores,
        dtype=float,
    ).copy()

    bayesian_score_dist = np.maximum(
        bayesian_score_dist,
        0.0,
    )

    if bayesian_score_dist.sum() <= 0:
        bayesian_score_dist = np.full(
            11,
            1.0 / 11.0,
            dtype=float,
        )
    else:
        bayesian_score_dist /= (
            bayesian_score_dist.sum()
        )

    # Existing business bucket probabilities.
    promoter_probability = float(
        final_dist[0]
    )
    passive_probability = float(
        final_dist[1]
    )
    detractor_probability = float(
        final_dist[2]
    )

    # Redistribute each business bucket across its
    # individual scores according to the Bayesian posterior.
    final_score_dist = np.zeros(
        11,
        dtype=float,
    )

    detractor_shape = (
        bayesian_score_dist[0:7]
    )
    passive_shape = (
        bayesian_score_dist[7:9]
    )
    promoter_shape = (
        bayesian_score_dist[9:11]
    )

    detractor_shape_sum = float(
        detractor_shape.sum()
    )
    passive_shape_sum = float(
        passive_shape.sum()
    )
    promoter_shape_sum = float(
        promoter_shape.sum()
    )

    if detractor_shape_sum > 0:
        final_score_dist[0:7] = (
            detractor_shape
            / detractor_shape_sum
            * detractor_probability
        )
    else:
        final_score_dist[0:7] = (
            detractor_probability
            / 7.0
        )

    if passive_shape_sum > 0:
        final_score_dist[7:9] = (
            passive_shape
            / passive_shape_sum
            * passive_probability
        )
    else:
        final_score_dist[7:9] = (
            passive_probability
            / 2.0
        )

    if promoter_shape_sum > 0:
        final_score_dist[9:11] = (
            promoter_shape
            / promoter_shape_sum
            * promoter_probability
        )
    else:
        final_score_dist[9:11] = (
            promoter_probability
            / 2.0
        )

    final_score_dist = np.maximum(
        final_score_dist,
        0.0,
    )

    final_score_dist /= (
        final_score_dist.sum()
    )

    # ---- 5A. Convert 0–10 distribution to survey counts ----

    raw_counts = (
        final_score_dist
        * total_surveys
    )

    counts = np.floor(
        raw_counts
    ).astype(int)

    # Largest-remainder allocation guarantees that
    # score counts sum exactly to total_surveys.
    remainder = (
        int(total_surveys)
        - int(counts.sum())
    )

    if remainder > 0:
        fractional = (
            raw_counts
            - counts
        )

        order = np.argsort(
            -fractional
        )

        for idx in order[:remainder]:
            counts[idx] += 1

    counts = np.maximum(
        counts,
        0,
    )

    # ---- 5B. Business buckets from actual 0–10 counts ----

    detractors = int(
        counts[0:7].sum()
    )

    passives = int(
        counts[7:9].sum()
    )

    promoters = int(
        counts[9:11].sum()
    )

    # ---- 6. NPS, hard-clamped to the guaranteed [75, 100] range ----
    nps = ((promoters - detractors) / total_surveys) * 100 if total_surveys > 0 else 0.0
    nps = float(np.clip(nps, -100.0, 100.0))

    if preds is not None and len(preds) > 1:
        arr = np.asarray(preds)
        std = np.std(arr[:, 0])
        confidence = np.clip(np.exp(-std / 10) * 100, 50, 99)
        ci = max(1.0, 1.96 * std)
    else:
        # Confidence from prediction distribution entropy
        prob = np.asarray(counts, dtype=float)
        prob = np.maximum(prob, 1e-12)
        prob = prob / (prob.sum() + 1e-9)
        entropy = -np.sum(prob * np.log(prob))
        max_entropy = np.log(len(prob))

        confidence = np.clip(
            (1 - entropy / max_entropy) * 100,
            50,
            99
        )

        ci = max(1.0, (100 - confidence) / 5)

    result = {
        # Bayesian posterior probability for each individual
        # NPS survey score from 0 through 10.
        "bayesian_score_distribution": {
            f"score_{i}": float(
                round(
                    final_score_dist[i],
                    8,
                )
            )
            for i in range(11)
        },

        # Predicted number of surveys at each score.
        "score_counts": {
            f"score_{i}": int(
                counts[i]
            )
            for i in range(11)
        },

        # Business bucket counts derived ONLY after
        # the individual score distribution is established.
        "promoters": int(promoters),
        "passives": int(passives),
        "detractors": int(detractors),
        "nps": round(nps, 2),
        "confidence": float(round(float(confidence), 1)),
        "prediction_interval": {
            "low": float(round(max(-100.0, nps - ci), 2)),
            "high": float(round(min(100.0, nps + ci), 2)),
        },
        "top_drivers": [],
    }
    # ----------------------------------------------------------
    # AXIPULSEAI: Bayesian + Monte Carlo operate on 0..10
    # ----------------------------------------------------------
    try:
        result = _axi_attach_0_10_probabilistic_analysis(
            result,
            total_surveys=total_surveys,
            observed_counts=None,
            simulations=1000,
        )
    except Exception as probabilistic_error:
        logger.warning(
            f"0-10 probabilistic analysis failed: {probabilistic_error}"
        )

    return result
def predict_single(predictor, X, row):
    # The trained model is ALWAYS called. A model failure or an output-shape
    # mismatch is surfaced as an exception; it is never silently replaced by a
    # heuristic/fallback prediction (see requirement: no silent fallback path).
    if hasattr(predictor, "ensemble_weights") and predictor.ensemble_weights:
        pred = np.zeros(11, dtype=float)

        for name, weight in predictor.ensemble_weights.items():
            model = predictor._all_models.get(name)
            if model is not None:
                p = np.asarray(model.predict(X)[0], dtype=float)
                p = np.maximum(p, 0)
                p = p / (p.sum() + 1e-9)
                pred += p * weight
    else:
        pred = np.asarray(predictor.model.predict(X)[0], dtype=float)

    if len(pred) != 11:
        raise RuntimeError(
            f"Selected NPS model '{getattr(predictor, 'model_name', None)}' "
            f"returned {len(pred)} outputs; expected 11."
        )
    return postprocess_predictions(pred, row)

def predict_ensemble(predictor, X, row):
    if not predictor._all_models:
        raise RuntimeError(
            "Ensemble prediction requested but no candidate models are "
            "available (predictor._all_models is empty)."
        )
    preds = []
    for name, model in tqdm(predictor._all_models.items(), desc="Predicting models", leave=False):
        p = np.asarray(model.predict(X)[0], dtype=float)
        if len(p) >= 11:
            preds.append(p)

    if not preds:
        raise RuntimeError(
            "Ensemble prediction produced no valid 11-output predictions."
        )

    if predictor.config.ensemble_weighted and predictor.algorithm_performance:
        weights = compute_ensemble_weights(predictor.algorithm_performance)
        avg_pred = np.zeros(11, dtype=float)
        for name, model in tqdm(predictor._all_models.items(), desc="Predicting models", leave=False):
            p = np.asarray(model.predict(X)[0], dtype=float)
            if len(p) >= 11:
                avg_pred += p * weights.get(name, 0)
        avg_pred = avg_pred / sum(weights.values())
    else:
        avg_pred = np.mean(preds, axis=0)

    result = postprocess_predictions(avg_pred, row, preds)
    if isinstance(result, dict) and "promoters" in result:
        return result
    raise RuntimeError(
        "Ensemble prediction did not produce a valid NPS result."
    )

def predict_leaderboard(predictor, row_data, history_buffer=None):
    row = row_data.copy()
    if "date" not in row:
        row["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    X = align_features(row, predictor.feature_names, predictor._feature_stats, history_buffer)

    results = {}
    failed = []
    for name, model in tqdm(predictor._all_models.items(), desc="Predicting models", leave=False):
        try:
            p = model.predict(X)[0]
            if len(p) >= 11:
                res = postprocess_predictions(p, row)
                if isinstance(res, dict) and "promoters" in res:
                    results[name] = res
                else:
                    failed.append(name)
            else:
                failed.append(name)
        except Exception:
            failed.append(name)

    # Ensemble from averaged probabilities
    if results:
        preds = []
        for name, model in tqdm(predictor._all_models.items(), desc="Predicting models", leave=False):
            try:
                p = model.predict(X)[0]
                if len(p) >= 11:
                    preds.append(p)
            except Exception:
                pass
        if preds:
            avg_pred = np.mean(preds, axis=0)
            ensemble_res = postprocess_predictions(avg_pred, row, preds)
            if isinstance(ensemble_res, dict) and "promoters" in ensemble_res:
                results["[ENSEMBLE]"] = ensemble_res
    return results, failed
