from tqdm import tqdm

"""
AxiPulseAI – Inference (11-score distribution)
"""
import numpy as np
import pandas as pd
import logging

from .feature_engineering import align_features
from .models import compute_ensemble_weights
from core.probabilistic.categorical_nps import attach_probabilistic_analysis

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
        # Degraded mode (no model artifact / no 0..10 score distribution): there
        # is no distribution-based Bayesian/Monte Carlo uncertainty source, so
        # the interval is a point-only, zero-width band. It must NEVER fabricate
        # a scalar-NPS ± confidence band.
        "prediction_interval": {"low": float(nps), "high": float(nps)},
        "top_drivers": []
    }




def postprocess_predictions(pred_pct, row, preds=None, strict=False):
    """
    Convert the model's 11-score output into the canonical probabilistic NPS flow.

    The trained model produces probabilities for survey scores 0..10.

    This function does NOT perform Bayesian inference itself.

    Canonical flow:

        ML 0..10 distribution
            -> canonical Bayesian/Monte Carlo layer
            -> survey-score counts
            -> promoter/passive/detractor counts
            -> NPS

    Bayesian and Monte Carlo uncertainty therefore operate on the individual
    survey-score distribution, never on an already-computed scalar NPS.
    """
    # ---- 1. Determine survey population ----
    total_calls = _row_float(
        row,
        "total_calls_received",
        100,
    )

    release_rate = _row_float(
        row,
        "actual_release_rate",
        60.0,
    )

    released_calls = max(
        1,
        int(round(total_calls * release_rate / 100)),
    )

    health = _row_float(
        row,
        "operational_health",
        80.0,
    )

    health_clamped = max(
        0,
        min(120, health),
    )

    survey_rate = (
        0.08
        + (health_clamped / 120) * 0.03
    )

    survey_rate = np.clip(
        survey_rate,
        0.08,
        0.11,
    )

    total_surveys = max(
        1,
        int(round(released_calls * survey_rate)),
    )

    # ---- 2. Validate and normalize ML 0..10 distribution ----
    scores = np.asarray(
        pred_pct,
        dtype=float,
    ).flatten()

    if scores.size != 11:
        raise RuntimeError(
            f"Expected 11 outputs, got {scores.size}"
        )

    scores = np.maximum(
        scores,
        0.0,
    )

    if scores.sum() <= 0.0:
        scores = np.full(
            11,
            1.0 / 11.0,
            dtype=float,
        )
    else:
        scores /= scores.sum()

    # ----------------------------------------------------------
    # 3. CANONICAL PROBABILISTIC NPS PATH
    #
    # inference.py does not perform Bayesian inference.
    #
    # categorical_nps.py owns:
    #
    #   ML 0..10 probabilities
    #          ↓
    #   Bayesian Dirichlet model
    #          ↓
    #   Monte Carlo survey sampling
    #          ↓
    #   0..10 survey counts
    #          ↓
    #   Promoter / Passive / Detractor
    #          ↓
    #   NPS
    #
    # NPS is therefore always derived from survey-score outcomes.
    # ----------------------------------------------------------

    result = {
        "bayesian_score_distribution": {
            f"score_{i}": float(scores[i])
            for i in range(11)
        },
        "top_drivers": [],
    }

    probabilistic_available = True
    try:
        result = attach_probabilistic_analysis(
            result,
            total_surveys=total_surveys,
            observed_counts=None,
            simulations=1000,
            seed=42,
        )
    except Exception as probabilistic_error:
        probabilistic_available = False
        logger.warning(
            "0-10 probabilistic analysis failed: %s",
            probabilistic_error,
        )

        # Preserve the deterministic ML score distribution if the
        # probabilistic layer cannot run. Do not implement a second
        # Bayesian/Monte Carlo path here.
        result["bayesian_score_distribution"] = {
            f"score_{i}": float(scores[i])
            for i in range(11)
        }
        result["probabilistic_error"] = str(probabilistic_error)

    # ----------------------------------------------------------
    # 4. Canonical score distribution -> survey counts
    #
    # Normally attach_probabilistic_analysis() has already generated
    # score_counts. The fallback below exists only to preserve a valid
    # deterministic result if the probabilistic layer fails.
    # ----------------------------------------------------------

    if "score_counts" not in result:
        raw_counts = scores * total_surveys

        counts = np.floor(
            raw_counts
        ).astype(int)

        remainder = (
            int(total_surveys)
            - int(counts.sum())
        )

        if remainder > 0:
            fractional = raw_counts - counts
            order = np.argsort(-fractional)

            for idx in order[:remainder]:
                counts[idx] += 1

        counts = np.maximum(
            counts,
            0,
        )

        result["score_counts"] = {
            f"score_{i}": int(counts[i])
            for i in range(11)
        }

    # ----------------------------------------------------------
    # 5. Derive business buckets ONLY from individual survey scores
    # ----------------------------------------------------------

    counts = np.asarray(
        [
            int(
                result["score_counts"].get(
                    f"score_{i}",
                    0,
                )
            )
            for i in range(11)
        ],
        dtype=int,
    )

    detractors = int(
        counts[0:7].sum()
    )

    passives = int(
        counts[7:9].sum()
    )

    promoters = int(
        counts[9:11].sum()
    )

    total_counted_surveys = int(
        counts.sum()
    )

    if total_counted_surveys > 0:
        nps = (
            (promoters - detractors)
            / total_counted_surveys
        ) * 100.0
    else:
        nps = 0.0

    nps = float(
        np.clip(
            nps,
            -100.0,
            100.0,
        )
    )

    # Keep canonical probabilistic uncertainty if supplied.
    if (
        "monte_carlo_nps_p05" in result
        and "monte_carlo_nps_p95" in result
    ):
        prediction_interval = {
            "low": float(
                result["monte_carlo_nps_p05"]
            ),
            "high": float(
                result["monte_carlo_nps_p95"]
            ),
        }
    else:
        prediction_interval = {
            "low": float(nps),
            "high": float(nps),
        }

    result.update(
        {
            "score_counts": {
                f"score_{i}": int(counts[i])
                for i in range(11)
            },
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "nps": round(nps, 2),
            "prediction_interval": prediction_interval,
            "top_drivers": result.get(
                "top_drivers",
                [],
            ),
        }
    )

    # Expose the probabilistic engine status explicitly. When the Bayesian/
    # Monte-Carlo layer failed, the result is a deterministic point value from
    # the ML 0..10 distribution and MUST NOT present itself as full probabilistic
    # uncertainty. This never fabricates a scalar-NPS Bayesian/Monte-Carlo
    # interval; the interval remains derived from the 0..10 survey distribution.
    result["probabilistic_uncertainty"] = (
        "available" if probabilistic_available else "degraded_unavailable"
    )

    return result

def predict_single_vector(predictor, X):
    """Produce the canonical 11-score distribution for a single feature row.

    This is the SINGLE source of truth for how a selected NPS model (or the
    persisted weighted ensemble) yields the raw 0..10 score vector BEFORE
    postprocessing. Single prediction, batch prediction and any other consumer
    MUST go through this helper so that for the same input and the same
    predictor configuration the resulting 11-score distribution is identical
    (single prediction semantics == batch prediction semantics).

    The trained model is ALWAYS called. A model failure or an output-shape
    mismatch is surfaced as an exception; it is never silently replaced by a
    heuristic/fallback prediction (see requirement: no silent fallback path).
    """
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
    return pred


def predict_single(predictor, X, row):
    # The trained model is ALWAYS called. A model failure or an output-shape
    # mismatch is surfaced as an exception; it is never silently replaced by a
    # heuristic/fallback prediction (see requirement: no silent fallback path).
    return postprocess_predictions(predict_single_vector(predictor, X), row)

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
