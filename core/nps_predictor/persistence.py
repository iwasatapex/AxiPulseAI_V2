"""
AxiPulseAI – Persistence (Save/Load)
"""
import joblib
import logging
from pathlib import Path
from .constants import MODEL_VERSION
from .config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

def save_model(predictor, filepath: str):
    data = {
        "engine_version": MODEL_VERSION,
        "model": predictor.model,
        "model_name": predictor.model_name,
        "feature_names": predictor.feature_names,
        "history_days": predictor.history_days,
        "algorithm_performance": predictor.algorithm_performance,
        "algorithm_bucket_mae": getattr(predictor, "algorithm_bucket_mae", {}),
        "trained": predictor.trained,
        "fallback_value": predictor._fallback_value,
        "all_models": predictor._all_models,
        "ensemble_weights": getattr(predictor, "ensemble_weights", {}),
        "metadata": predictor.metadata,
        "tuned_params": predictor.tuned_params,
        "config": predictor.config,
        "feature_importance": predictor._feature_importance,
        "feature_stats": predictor._feature_stats,
        "target_means": predictor._target_means,
        "history_buffer": predictor._history_buffer,
    }
    joblib.dump(data, filepath, compress=3)
    logger.info(f"AxiPulseAI model saved to {filepath}")

def load_model(predictor, filepath: str):
    data = joblib.load(filepath)

    saved_version = data.get("engine_version", "unknown")
    if saved_version != MODEL_VERSION:
        logger.warning(
            f"Saved model version {saved_version} differs from current {MODEL_VERSION}."
        )

    predictor.model = data.get("model")
    predictor.model_name = data.get("model_name")
    predictor.feature_names = data.get("feature_names", [])
    predictor.history_days = data.get("history_days", 0)
    predictor.algorithm_performance = data.get("algorithm_performance", {})
    predictor.algorithm_bucket_mae = data.get("algorithm_bucket_mae", {})
    predictor.trained = data.get("trained", False)
    predictor._fallback_value = data.get("fallback_value", 100.0)
    predictor._all_models = data.get("all_models", {})
    predictor.ensemble_weights = data.get("ensemble_weights", {})
    predictor.metadata = data.get("metadata", {})
    predictor.tuned_params = data.get("tuned_params", {})
    predictor._feature_importance = data.get("feature_importance", {})
    predictor._feature_stats = data.get("feature_stats", {})
    predictor._target_means = data.get("target_means", None)
    predictor._history_buffer = data.get("history_buffer", None)
    predictor._shap_explainer = None

    loaded_config = data.get("config", None)
    if loaded_config is not None:
        default_vars = vars(DEFAULT_CONFIG)
        for k, v in default_vars.items():
            if not hasattr(loaded_config, k):
                setattr(loaded_config, k, v)
        predictor.config = loaded_config
    else:
        predictor.config = DEFAULT_CONFIG

    if predictor.model is not None and hasattr(predictor.model, "n_features_in_"):
        expected = predictor.model.n_features_in_
        actual = len(getattr(predictor, "feature_names", []))

        # Legacy Engine2 models may report 0 features.
        if expected > 0 and actual > 0 and expected != actual:
            logger.warning(
                f"Feature count mismatch: model expects {expected}, "
                f"but we have {actual}."
            )

    logger.info(f"AxiPulseAI model loaded from {filepath}")
