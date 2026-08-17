"""
AxiPulseAI – Explainability (SHAP)
Fixed for CatBoost native multi‑output
"""
import numpy as np
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .feature_engineering import align_features

logger = logging.getLogger(__name__)

def compute_shap(predictor, X):
    """Compute SHAP explainer – works with CatBoost, XGBoost, LightGBM, and sklearn."""
    if not SHAP_AVAILABLE or predictor.model is None:
        return None

    try:
        X_sample = X.sample(min(100, len(X)), random_state=predictor.config.random_state)
        actual_model = predictor.model

        # For MultiOutputRegressor, take the first estimator (promoter model)
        if hasattr(actual_model, 'estimators_'):
            actual_model = actual_model.estimators_[0]
        elif hasattr(actual_model, 'estimator_'):
            actual_model = actual_model.estimator_

        # ---- CatBoost special handling ----
        # CatBoost native multi‑output has no `estimators_`; it's a single model.
        # Use `shap.Explainer` with a custom predict function.
        if 'catboost' in str(type(actual_model)).lower():
            # CatBoost's predict returns (n_samples, n_outputs)
            def catboost_predict(X):
                return actual_model.predict(X)

            explainer = shap.Explainer(catboost_predict, X_sample, feature_names=X_sample.columns)
            predictor._shap_explainer = explainer
            logger.info("SHAP explainer (CatBoost) computed.")
            return explainer

        # ---- Tree models (XGBoost, LightGBM, sklearn) ----
        if hasattr(actual_model, 'tree_') or hasattr(actual_model, 'get_booster'):
            explainer = shap.TreeExplainer(
                actual_model,
                feature_perturbation="tree_path_dependent"
            )
        elif hasattr(shap, 'Explainer'):
            explainer = shap.Explainer(actual_model, X_sample)
        else:
            logger.warning("SHAP not available for this model type.")
            return None

        predictor._shap_explainer = explainer
        logger.info("SHAP explainer computed.")
        return explainer

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None

def explain_nps(predictor, row_data, history_buffer=None):
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not installed. Run: pip install shap"}

    if predictor._shap_explainer is None:
        return {"error": "SHAP explainer not available. Train first or call compute_shap."}

    X = align_features(row_data, predictor.feature_names, predictor._feature_stats, history_buffer)

    try:
        # Get SHAP values
        shap_values = predictor._shap_explainer.shap_values(X)

        # Handle different return types
        if isinstance(shap_values, list):
            # Multi‑output: take first (promoters)
            shap_promoter = shap_values[0].flatten()
            expected = np.asarray(predictor._shap_explainer.expected_value).flatten()
            if len(expected) > 1:
                expected = expected[0]  # take first

            return {
                "base_value": float(expected),
                "shap_values": {
                    "promoter": dict(zip(predictor.feature_names, shap_promoter)),
                },
                "prediction": {
                    "promoter_pct": float(predictor.model.predict(X)[0][0]),
                    "passive_pct": float(predictor.model.predict(X)[0][1]),
                }
            }
        else:
            # Single output
            shap_vals = shap_values.flatten()
            expected = np.asarray(predictor._shap_explainer.expected_value).flatten()[0]
            return {
                "base_value": float(expected),
                "shap_values": dict(zip(predictor.feature_names, shap_vals)),
                "prediction": {
                    "promoter_pct": float(predictor.model.predict(X)[0][0]),
                    "passive_pct": float(predictor.model.predict(X)[0][1]),
                }
            }

    except Exception as e:
        return {"error": str(e)}
