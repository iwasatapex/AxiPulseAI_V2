"""
Model save/load.

Moved verbatim out of OperationalHealthPredictor in operation_health_predictor.py
(Phase 2, Step 5 — no logic changed). Old saved models keep loading
correctly: the joblib payload keys/shape are unchanged.
"""

from pathlib import Path

import joblib

from .config import DEFAULT_CONFIG
from .constants import MODEL_VERSION
from .utils import TF_AVAILABLE, keras, logger


class PersistenceMixin:
    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------
    def save_model(self, filepath):
        tf_model_path = None
        if self._tf_model is not None:
            tf_model_path = str(Path(filepath).with_suffix('.keras'))
            self._tf_model.save(tf_model_path)

        all_models_to_save = {k: v for k, v in self._all_models.items() if k != "TensorFlow"}

        data = {
            "model": self.model,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "history_days": self.history_days,
            "algorithm_performance": self.algorithm_performance,
            "trained": self.trained,
            "fallback_value": self._fallback_value,
            "all_models": all_models_to_save,
            "metadata": self.metadata,
            "tuned_params": self.tuned_params,
            "config": self.config,
            "feature_importance": self._feature_importance,
            "tf_scaler": self._tf_scaler,
            "issue_cols": self._issue_cols,
            "tf_model_path": tf_model_path,
            "feature_stats": self._feature_stats,
        }
        joblib.dump(data, filepath, compress=3)
        logger.info(f"Model v{MODEL_VERSION} saved to {filepath}")

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.model = data.get("model")
        self.model_name = data.get("model_name")
        self.feature_names = data.get("feature_names", [])
        self.history_days = data.get("history_days", 0)
        self.algorithm_performance = data.get("algorithm_performance", {})
        self.trained = data.get("trained", False)
        self._fallback_value = data.get("fallback_value", 100.0)
        self._all_models = data.get("all_models", {})
        self.metadata = data.get("metadata", {})
        self.tuned_params = data.get("tuned_params", {})
        self.config = data.get("config", DEFAULT_CONFIG)
        self._feature_importance = data.get("feature_importance", {})
        self._tf_scaler = data.get("tf_scaler", None)
        self._issue_cols = data.get("issue_cols", [])
        self._feature_stats = data.get("feature_stats", {})

        # Backfill any config fields missing from older saved models so the
        # final-fit resource guard uses sensible defaults instead of raising
        # AttributeError on a legacy payload.
        loaded_config = data.get("config", None)
        if loaded_config is not None:
            for k, v in vars(DEFAULT_CONFIG).items():
                if not hasattr(loaded_config, k):
                    setattr(loaded_config, k, v)
            self.config = loaded_config
        else:
            self.config = DEFAULT_CONFIG

        tf_model_path = data.get("tf_model_path")
        if tf_model_path and Path(tf_model_path).exists() and TF_AVAILABLE:
            try:
                self._tf_model = keras.models.load_model(tf_model_path)
                logger.info("TensorFlow model loaded.")
                if self.model_name == "TensorFlow":
                    self.model = self._tf_model
                    self._all_models["TensorFlow"] = self._tf_model
            except Exception as e:
                logger.warning(f"Failed to load TensorFlow model: {e}")

        if self.model is not None:
            expected = None

            if hasattr(self.model, "feature_count_"):
                expected = self.model.feature_count_
            elif hasattr(self.model, "n_features_in_"):
                expected = self.model.n_features_in_

            if expected and len(self.feature_names) != expected:
                logger.warning(
                    f"Feature count mismatch: model expects {expected}, "
                    f"but we have {len(self.feature_names)}. Predictions may be wrong."
                )

        self._clear_reverse_cache()
        logger.info(f"Model v{self.metadata.get('engine_version', 'unknown')} loaded from {filepath}")


def save_model(predictor, filepath):
    """Compatibility API for PersistenceMixin.save_model()."""
    if predictor is None:
        raise TypeError(
            "save_model() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return PersistenceMixin.save_model(
        predictor,
        filepath,
    )


def load_model(predictor, filepath):
    """Compatibility API for PersistenceMixin.load_model()."""
    if predictor is None:
        raise TypeError(
            "load_model() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return PersistenceMixin.load_model(
        predictor,
        filepath,
    )
