"""
OperationalHealthPredictor — public entry point.

Phase 2, Step 8: predictor.py is shrunk down to just the class
definition (constructor) plus the mixins that supply its behavior.
Every method's logic still lives verbatim in its own module; this file
only wires them together via multiple inheritance so
`OperationalHealthPredictor` behaves exactly as it did as one big class.
"""

from typing import Optional

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .feature_engineering import FeatureEngineeringMixin
from .inference import InferenceMixin
from .models import ModelRegistryMixin
from .persistence import PersistenceMixin
from .preprocessing import DataLoadingMixin
from .trainer import TrainerMixin
from .utils import TF_AVAILABLE, tf
from .validation import ValidationMixin


class OperationalHealthPredictor(
    ValidationMixin,
    DataLoadingMixin,
    FeatureEngineeringMixin,
    ModelRegistryMixin,
    TrainerMixin,
    InferenceMixin,
    PersistenceMixin,
):
    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.model_name = None
        self.feature_names = []
        self.history_days = 0
        self.algorithm_performance = {}
        self._all_models = {}
        self._fallback_value = 100.0
        self.trained = False
        self.metadata = {}
        self.history_file = None
        self.tuned_params = {}
        self._feature_importance = None
        self._shap_explainer = None
        self._issue_cols = []
        self._feature_stats = {}
        self._train_mean = None
        self._train_std = None

        self._tf_model = None
        self._tf_scaler = None

        self._rng = np.random.default_rng(self.config.random_state)

        if TF_AVAILABLE:
            tf.random.set_seed(self.config.random_state)
