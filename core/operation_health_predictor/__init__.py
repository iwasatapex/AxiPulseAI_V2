"""
AxiPulseAI – Core Engine (v10.10)

Package layout (Phase 1/2 refactor of the original single-file
operation_health_predictor.py — no behavior changes):

    operation_health_predictor/
        constants.py           RELEASE_TARGET, DEFAULT_BOUNDS, REQUIRED_COLUMNS, ...
        config.py               Config dataclass, DEFAULT_CONFIG
        utils.py                 logging setup, reproducibility seed, optional-dep imports
        validation.py            ValidationMixin
        preprocessing.py         DataLoadingMixin (load_data, feature stats)
        feature_engineering.py   FeatureEngineeringMixin (prepare_features, alignment)
        models.py                ModelRegistryMixin (model factory, TF model builder)
        trainer.py               TrainerMixin (train / cold-start / rolling-origin / importance)
        inference.py             InferenceMixin (predict / leaderboard / reverse-optimize / SHAP)
        persistence.py           PersistenceMixin (save_model / load_model)
        predictor.py             OperationalHealthPredictor (composes all of the above)

`operation_health_predictor.py` at the repo root is kept as a thin compatibility
wrapper that re-exports everything below, so existing imports like
`from core.operation_health_predictor import OperationalHealthPredictor` keep working.
"""

from .config import Config, DEFAULT_CONFIG
from .constants import (
    ALLOWED_OPTIMIZE_FACTORS,
    DEFAULT_BOUNDS,
    ISSUE_PREFIX,
    MODEL_VERSION,
    RELEASE_TARGET,
    REQUIRED_COLUMNS,
    TRANSFER_TARGET,
)
from .predictor import OperationalHealthPredictor

__all__ = [
    "OperationalHealthPredictor",
    "Config",
    "DEFAULT_CONFIG",
    "RELEASE_TARGET",
    "TRANSFER_TARGET",
    "ISSUE_PREFIX",
    "MODEL_VERSION",
    "DEFAULT_BOUNDS",
    "ALLOWED_OPTIMIZE_FACTORS",
    "REQUIRED_COLUMNS",
]

from .probabilistic import adapt_oh_prediction, adapt_oh_result
