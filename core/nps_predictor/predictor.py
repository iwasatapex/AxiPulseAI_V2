import gc

"""
AxiPulseAI – Main Predictor Class (distribution mode)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from .config import Config, DEFAULT_CONFIG
from .trainer import train_nps_predictor
from .inference import predict_single, predict_ensemble, predict_leaderboard
from .explainability import explain_nps, compute_shap
from .optimizer import reverse_optimize_nps
from .persistence import save_model, load_model
from .validation import detect_drift, needs_retraining
from .feature_engineering import align_features

logger = logging.getLogger(__name__)

class NPSPredictor:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.model_name = None
        self.feature_names = []
        self.history_days = 0
        self.algorithm_performance = {}  # primary leaderboard metric: NPS MAE per model
        self.algorithm_bucket_mae = {}   # secondary diagnostic: raw bucket-count MAE per model
        self._all_models = {}
        self._fallback_value = 100.0
        self.trained = False
        self.metadata = {}
        self.history_file = None
        self.tuned_params = {}
        self._feature_importance = None
        self._shap_explainer = None
        self._feature_stats = {}
        self._target_means = None
        self._history_buffer = None
        self._rng = np.random.default_rng(self.config.random_state)

    def train(self, filepath: Union[str, Path], tune=False, progress=None):
        train_nps_predictor(self, str(filepath), tune, progress=progress)

    def _compute_feature_importance(self):
        if self.model is not None:
            try:
                if hasattr(self.model, 'feature_importances_'):
                    importances = self.model.feature_importances_
                    if len(importances) == len(self.feature_names):
                        self._feature_importance = dict(zip(self.feature_names, importances))
                        return
                elif hasattr(self.model, 'coef_'):
                    coefs = self.model.coef_.flatten()
                    if len(coefs) == len(self.feature_names):
                        self._feature_importance = dict(zip(self.feature_names, np.abs(coefs)))
                        return
                if hasattr(self.model, 'estimators_'):
                    importances_list = []
                    for est in self.model.estimators_:
                        if hasattr(est, 'feature_importances_'):
                            imp = est.feature_importances_
                            if len(imp) == len(self.feature_names):
                                importances_list.append(imp)
                    if importances_list:
                        avg_imp = np.mean(importances_list, axis=0)
                        self._feature_importance = dict(zip(self.feature_names, avg_imp))
                        return
                self._feature_importance = {}
            except Exception as e:
                logger.warning(f"Could not compute feature importance: {e}")
                self._feature_importance = {}

    def predict(self, row_data: dict, history_buffer: Optional[pd.DataFrame] = None, use_ensemble: Optional[bool] = None) -> Dict:
        if not self.trained:
            raise RuntimeError("Model not trained.")
        row = row_data.copy()
        if "date" not in row:
            row["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        buffer = history_buffer if history_buffer is not None else self._history_buffer
        X = align_features(row, self.feature_names, self._feature_stats, buffer)

        if use_ensemble is None:
            use_ensemble = self.config.use_ensemble or (self.model is None)

        if use_ensemble and self._all_models:
            result = predict_ensemble(self, X, row)
        else:
            result = predict_single(self, X, row)

        return result

    def predict_leaderboard(self, row_data: dict, history_buffer: Optional[pd.DataFrame] = None):
        return predict_leaderboard(self, row_data, history_buffer)

    def explain(self, row_data: dict, history_buffer: Optional[pd.DataFrame] = None):
        return explain_nps(self, row_data, history_buffer)

    def reverse_optimize_nps(self, target_nps, optimize_factors, fixed_values, total_calls):
        # This still uses the old optimizer; may need adaptation.
        return reverse_optimize_nps(self, target_nps, optimize_factors, fixed_values, total_calls)

    def check_drift(self, new_data: pd.DataFrame) -> Dict:
        drift = detect_drift(self._feature_stats, new_data)
        return {
            "drift_scores": drift,
            "needs_retraining": needs_retraining(drift),
            "threshold": 0.2,
        }

    def get_feature_importance(self):
        if not self.trained:
            raise RuntimeError("Model not trained.")
        return dict(sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True))

    def save_model(self, filepath: str):
        save_model(self, filepath)

    def load_model(self, filepath: str):
        load_model(self, filepath)
