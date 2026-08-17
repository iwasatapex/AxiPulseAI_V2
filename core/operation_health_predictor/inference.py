"""
Prediction, leaderboard, reverse-optimization, feature importance, and
SHAP explanation.

Moved verbatim out of OperationalHealthPredictor in operation_health_predictor.py
(Phase 2, Step 6 — no logic changed).
"""

from datetime import datetime
from functools import lru_cache

import numpy as np
from scipy.optimize import differential_evolution, minimize
from sklearn.pipeline import Pipeline

from .constants import ALLOWED_OPTIMIZE_FACTORS, DEFAULT_BOUNDS
from .utils import SHAP_AVAILABLE, logger, shap


class InferenceMixin:
    # ---------------------------------------------------------------
    # Prediction (no clipping)
    # ---------------------------------------------------------------
    def predict(self, row_data, apply_oif=False, history_buffer=None):
        if not self.trained:
            raise RuntimeError("Model not trained.")
        valid, errors = self.validate_prediction(row_data)
        if not valid:
            if self.config.strict_prediction:
                raise ValueError("; ".join(errors))
            else:
                logger.warning(f"Validation errors: {errors}")
        row = row_data.copy()
        if "date" not in row:
            row["date"] = datetime.now()

        X = self._align_features(row, history_buffer)

        if self.model is None and not self._all_models:
            raise RuntimeError("No models available for prediction. Training may have failed.")

        if self.model is not None:
            try:
                if self.model_name == "TensorFlow" and self._tf_scaler is not None:
                    X_scaled = self._tf_scaler.transform(X)
                    pred_arr = self.model.predict(X_scaled)
                    prediction = float(np.asarray(pred_arr).ravel()[0])
                elif isinstance(self.model, Pipeline):
                    prediction = float(self.model.predict(X)[0])
                else:
                    prediction = float(self.model.predict(X)[0])
                if not np.isfinite(prediction):
                    logger.warning("Model returned non-finite value, using fallback.")
                    prediction = self._fallback_value
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                prediction = self._fallback_value
        else:
            # Ensemble
            if not self._all_models:
                raise RuntimeError("Ensemble is empty. No models to predict.")
            else:
                if self.config.ensemble_weighted and self.algorithm_performance:
                    mae_list = [self.algorithm_performance.get(name, 1) for name in self._all_models.keys()]
                    weights = [1.0/(m + 1e-6) for m in mae_list]
                    wsum = sum(weights)
                    weights = [w/wsum for w in weights]
                    weighted_sum = 0.0
                    for i, (name, model) in enumerate(self._all_models.items()):
                        try:
                            if name == "TensorFlow" and self._tf_scaler is not None:
                                X_scaled = self._tf_scaler.transform(X)
                                pred_arr = model.predict(X_scaled)
                                p = float(np.asarray(pred_arr).ravel()[0])
                            elif isinstance(model, Pipeline):
                                p = float(model.predict(X)[0])
                            else:
                                p = float(model.predict(X)[0])
                            if np.isfinite(p):
                                weighted_sum += p * weights[i]
                        except Exception as e:
                            logger.warning(f"Ensemble model {name} failed: {e}")
                    prediction = weighted_sum
                else:
                    preds = []
                    for name, model in self._all_models.items():
                        try:
                            if name == "TensorFlow" and self._tf_scaler is not None:
                                X_scaled = self._tf_scaler.transform(X)
                                pred_arr = model.predict(X_scaled)
                                p = float(np.asarray(pred_arr).ravel()[0])
                            elif isinstance(model, Pipeline):
                                p = float(model.predict(X)[0])
                            else:
                                p = float(model.predict(X)[0])
                            if np.isfinite(p):
                                preds.append(p)
                        except Exception as e:
                            logger.warning(f"{name} prediction failed: {e}")
                    prediction = float(np.mean(preds)) if preds else self._fallback_value

        if not np.isfinite(prediction):
            logger.warning("Prediction is non-finite, using fallback.")
            prediction = self._fallback_value

        # No clipping – return as‑is
        return float(prediction)

    # ---------------------------------------------------------------
    # Leaderboard (no clipping)
    # ---------------------------------------------------------------
    def predict_leaderboard(self, row_data, apply_oif=False, history_buffer=None):
        if not self.trained:
            raise RuntimeError("Model not trained.")
        valid, errors = self.validate_prediction(row_data)
        if not valid:
            if self.config.strict_prediction:
                raise ValueError("; ".join(errors))
            else:
                logger.warning(f"Validation errors: {errors}")

        row = row_data.copy()
        if "date" not in row:
            row["date"] = datetime.now()

        X = self._align_features(row, history_buffer)

        results = {}
        failed = []

        for name, model in self._all_models.items():
            try:
                if name == "TensorFlow" and self._tf_scaler is not None:
                    X_scaled = self._tf_scaler.transform(X)
                    pred_arr = model.predict(X_scaled)
                    p = float(np.asarray(pred_arr).ravel()[0])
                elif isinstance(model, Pipeline):
                    p = float(model.predict(X)[0])
                else:
                    p = float(model.predict(X)[0])
                if np.isfinite(p):
                    results[name] = p
                else:
                    failed.append(name)
            except Exception as e:
                logger.warning(f"Leaderboard: {name} failed -> {e}")
                failed.append(name)

        if results:
            if self.config.ensemble_weighted and self.algorithm_performance:
                mae_list = [self.algorithm_performance.get(name, 1) for name in results.keys()]
                weights = [1.0/(m + 1e-6) for m in mae_list]
                wsum = sum(weights)
                weights = [w/wsum for w in weights]
                weighted_sum = sum(p * w for p, w in zip(results.values(), weights))
                results["[ENSEMBLE]"] = weighted_sum
            else:
                results["[ENSEMBLE]"] = np.mean(list(results.values()))

        if self.model is not None:
            try:
                if self.model_name == "TensorFlow" and self._tf_scaler is not None:
                    X_scaled = self._tf_scaler.transform(X)
                    pred_arr = self.model.predict(X_scaled)
                    p = float(np.asarray(pred_arr).ravel()[0])
                elif isinstance(self.model, Pipeline):
                    p = float(self.model.predict(X)[0])
                else:
                    p = float(self.model.predict(X)[0])
                if np.isfinite(p) and p not in results.values():
                    results["[SELECTED]"] = p
            except Exception:
                pass

        return dict(sorted(results.items(), key=lambda item: item[1], reverse=True)), failed

    # ---------------------------------------------------------------
    # Reverse Optimizer
    # ---------------------------------------------------------------
    @lru_cache(maxsize=64)
    def _reverse_optimize_uncached(self, target_score, optimize_factors_tuple, fixed_values_tuple):
        optimize_factors = list(optimize_factors_tuple)
        fixed_values = dict(fixed_values_tuple)

        if not self.trained:
            raise RuntimeError("Model not trained.")
        if len(optimize_factors) < 1 or len(optimize_factors) > 3:
            raise ValueError("Must optimize 1-3 factors.")
        invalid = set(optimize_factors) - ALLOWED_OPTIMIZE_FACTORS
        if invalid:
            raise ValueError(f"Invalid factor(s): {invalid}. Allowed: {sorted(ALLOWED_OPTIMIZE_FACTORS)}")

        # FIX 1: Correctly check for stored median key in _feature_stats
        for col in self.feature_names:
            if col not in fixed_values and col not in optimize_factors:
                median_key = f"{col}_median"
                if median_key in self._feature_stats:
                    fixed_values[col] = float(self._feature_stats[median_key])
                else:
                    fixed_values[col] = 50.0

        # FIX 3: Construct optimization bounds safely using DEFAULT_BOUNDS
        bounds = []
        for f in optimize_factors:
            low, high = DEFAULT_BOUNDS.get(f, (0, 100))
            bounds.append((low, high))

        result = differential_evolution(
            lambda vector: objective(self, target_score, fixed_values, optimize_factors, vector),
            bounds,
            strategy="best1bin",
            maxiter=self.config.de_maxiter,
            popsize=self.config.de_popsize,
            tol=self.config.de_tol,
            seed=self.config.random_state,
            disp=False,
        )

        refined = minimize(
            lambda vector: objective(self, target_score, fixed_values, optimize_factors, vector),
            result.x,
            method="Nelder-Mead",
            options={"maxiter": self.config.nm_maxiter, "xatol": self.config.nm_xatol, "fatol": self.config.nm_fatol},
        )
        best_vector = refined.x if refined.success else result.x
        optimized_values = vector_to_dict(optimize_factors, best_vector)

        full_check = make_input_vector(fixed_values, optimized_values)
        valid, _ = self.validate_prediction(full_check)
        if not valid:
            for f in optimize_factors:
                lo, hi = DEFAULT_BOUNDS.get(f, (0, 100))
                optimized_values[f] = np.clip(optimized_values[f], lo, hi)

        final_pred = self.predict(make_input_vector(fixed_values, optimized_values), apply_oif=False)
        leaderboard, failed = self.predict_leaderboard(make_input_vector(fixed_values, optimized_values), apply_oif=False)

        return {
            "optimized_factors": optimized_values,
            "predicted_score": final_pred,
            "target_score": target_score,
            "error": abs(final_pred - target_score),
            "success": refined.success,
            "iterations": result.nit + refined.nit,
            "message": f"DE: {result.message}, Refine: {refined.message}",
            "leaderboard": leaderboard,
            "failed_models": failed,
        }

    def reverse_optimize(self, target_score, optimize_factors, fixed_values):
        optimize_factors_tuple = tuple(sorted(optimize_factors))
        fixed_values_tuple = tuple(sorted((k, float(v)) for k, v in fixed_values.items()))
        return self._reverse_optimize_uncached(target_score, optimize_factors_tuple, fixed_values_tuple)

    def _clear_reverse_cache(self):
        self._reverse_optimize_uncached.cache_clear()

    # ---------------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------------
    def _compute_shap(self, X):
        if not SHAP_AVAILABLE:
            self._shap_explainer = None
            return
        if self.model is None:
            self._shap_explainer = None
            return
        try:
            X_sample = X.sample(min(100, len(X)), random_state=self.config.random_state)
            actual_model = self.model
            if isinstance(actual_model, Pipeline):
                scaler = actual_model.named_steps.get('scaler')
                mlp = actual_model.named_steps.get('mlp')
                if mlp is not None and scaler is not None:
                    X_scaled = scaler.transform(X_sample)
                    self._shap_explainer = shap.Explainer(mlp, X_scaled)
                    logger.info("SHAP (Explainer) for MLP computed.")
                    return
            if hasattr(actual_model, 'tree_') or hasattr(actual_model, 'get_booster'):
                self._shap_explainer = shap.TreeExplainer(actual_model)
                logger.info("SHAP (Tree) computed.")
            else:
                if hasattr(shap, 'Explainer'):
                    self._shap_explainer = shap.Explainer(actual_model, X_sample)
                    logger.info("SHAP (Explainer) computed.")
                else:
                    logger.warning("SHAP Explainer not available; skipping SHAP.")
                    self._shap_explainer = None
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            self._shap_explainer = None

    def explain(self, row_data):
        if not self.trained:
            raise RuntimeError("Model not trained.")
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not installed. Run: pip install shap"}
        if self._shap_explainer is None:
            return {"error": "SHAP explainer not available. Try training on more data."}

        X = self._align_features(row_data)
        try:
            shap_values = self._shap_explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            expected = np.asarray(self._shap_explainer.expected_value).flatten()[0]
            pred = float(self.model.predict(X)[0])
            return {
                "base_value": float(expected),
                "shap_values": dict(zip(self.feature_names, shap_values.flatten())),
                "prediction": pred,
            }
        except Exception as e:
            return {"error": str(e)}


def make_input_vector(fixed_values, factor_values):
    """Merge fixed feature values with candidate factor values."""
    full = fixed_values.copy()
    full.update(factor_values)
    return full


def vector_to_dict(optimize_factors, vector):
    """Map an optimization vector back to factor-name keys."""
    return {optimize_factors[i]: float(vector[i]) for i in range(len(optimize_factors))}


def cost(predictor, target_score, fixed_values, factor_values):
    """Distance between the target score and the model's prediction."""
    try:
        pred = predictor.predict(make_input_vector(fixed_values, factor_values), apply_oif=False)
        return abs(target_score - pred)
    except Exception:
        return 1e9


def realism_penalty(predictor, fixed_values, optimize_factors, factor_values):
    """Quadratic drift penalty away from typical (IQR) factor ranges."""
    penalty = 0.0
    feature_stats = getattr(predictor, "_feature_stats", None) or {}
    for f in optimize_factors:
        current = fixed_values.get(f, 0)
        target = factor_values[f]

        default_low, default_high = DEFAULT_BOUNDS.get(f, (0, 100))
        q1 = feature_stats.get(f"{f}_q1", default_low)
        q3 = feature_stats.get(f"{f}_q3", default_high)

        low = max(q1, default_low)
        high = min(q3, default_high)
        range_scale = (high - low) if (high - low) > 0 else 1.0
        penalty += ((target - current) / range_scale) ** 2
    return penalty * 0.05


def objective(predictor, target_score, fixed_values, optimize_factors, vector):
    """Combined objective: prediction error plus realism penalty."""
    values = vector_to_dict(optimize_factors, vector)
    return cost(predictor, target_score, fixed_values, values) + realism_penalty(predictor, fixed_values, optimize_factors, values)


def predict(predictor, row_data, apply_oif=False, history_buffer=None):
    """Compatibility API for InferenceMixin.predict()."""
    if predictor is None:
        raise TypeError(
            "predict() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return InferenceMixin.predict(
        predictor,
        row_data,
        apply_oif=apply_oif,
        history_buffer=history_buffer,
    )


def predict_leaderboard(predictor, *args, **kwargs):
    """Compatibility API for InferenceMixin.predict_leaderboard()."""
    if predictor is None:
        raise TypeError(
            "predict_leaderboard() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return InferenceMixin.predict_leaderboard(
        predictor,
        *args,
        **kwargs,
    )


def reverse_optimize(predictor, *args, **kwargs):
    """Compatibility API for InferenceMixin.reverse_optimize()."""
    if predictor is None:
        raise TypeError(
            "reverse_optimize() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return InferenceMixin.reverse_optimize(
        predictor,
        *args,
        **kwargs,
    )


def explain(predictor, *args, **kwargs):
    """Compatibility API for InferenceMixin.explain()."""
    if predictor is None:
        raise TypeError(
            "explain() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return InferenceMixin.explain(
        predictor,
        *args,
        **kwargs,
    )
