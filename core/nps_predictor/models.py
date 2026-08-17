"""
AxiPulseAI – Full Model Registry (all models, multi‑output)
"""
import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

def create_model_registry(config, cold_start=False, history_days=0, num_outputs=11):
    v = 1 if config.verbose else 0

    # MLP pipeline with scaling
    mlp_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=config.mlp_hidden_layers,
            activation=config.mlp_activation,
            solver=config.mlp_solver,
            max_iter=config.mlp_max_iter,
            random_state=config.random_state,
            early_stopping=config.mlp_early_stopping,
            validation_fraction=config.mlp_validation_fraction,
            n_iter_no_change=config.mlp_n_iter_no_change,
            verbose=v,
        ))
    ])

    rf_kwargs = {
        "n_estimators": config.n_estimators,
        "random_state": config.random_state,
        "n_jobs": 4,
        "verbose": v,
    }
    if hasattr(RandomForestRegressor, "warm_start"):
        rf_kwargs["warm_start"] = True

    et_kwargs = rf_kwargs.copy()
    gb_kwargs = {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "random_state": config.random_state,
        "verbose": v,
        "n_iter_no_change": 10,
        "validation_fraction": 0.1,
        "tol": 1e-4,
    }
    hgb_kwargs = {
        "max_iter": config.n_estimators,
        "random_state": config.random_state,
        "verbose": v,
        "early_stopping": True,
        "n_iter_no_change": 10,
    }

    models = {}

    def safe_add(name, model_factory):
        try:
            models[name] = model_factory()
        except Exception as e:
            if config.verbose:
                print(f"⚠️ Skipping {name}: {e}")

    # CatBoost (supports multi-output natively)
    if CatBoostRegressor is not None:
        if config.use_catboost_multi:
            safe_add("CatBoost", lambda: CatBoostRegressor(
                iterations=min(config.n_estimators,500),
                learning_rate=config.learning_rate,
                random_seed=config.random_state,
                verbose=config.verbose,
                thread_count=1,
                loss_function="MultiRMSE",
                early_stopping_rounds=config.cat_early_stopping_rounds,
            ))
        else:
            safe_add("CatBoost", lambda: MultiOutputRegressor(
                CatBoostRegressor(
                    iterations=min(config.n_estimators,500),
                    learning_rate=config.learning_rate,
                    random_seed=config.random_state,
                    verbose=config.verbose,
                    thread_count=1,
                    early_stopping_rounds=config.cat_early_stopping_rounds,
                )
            ))

    if LGBMRegressor is not None:
        safe_add("LightGBM", lambda: MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                random_state=config.random_state,
                verbose=-1 if not v else v,
                n_jobs=4,
            )
        ))

    if XGBRegressor is not None:
        safe_add("XGBoost", lambda: MultiOutputRegressor(
            XGBRegressor(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                random_state=config.random_state,
                verbosity=1 if v else 0,
                n_jobs=4,
            )
        ))

    safe_add("ExtraTrees", lambda: MultiOutputRegressor(ExtraTreesRegressor(**et_kwargs)))
    safe_add("RandomForest", lambda: MultiOutputRegressor(RandomForestRegressor(**rf_kwargs)))
    safe_add("HistGradientBoosting", lambda: MultiOutputRegressor(HistGradientBoostingRegressor(**hgb_kwargs)))
    safe_add("GradientBoosting", lambda: MultiOutputRegressor(GradientBoostingRegressor(**gb_kwargs)))
    safe_add("MLP", lambda: mlp_pipeline)

    return models

def compute_ensemble_weights(performance: dict) -> dict:
    if not performance:
        return {}
    mae_values = np.array(list(performance.values()))
    weights = 1.0 / (mae_values + 1e-6)
    weights = weights / np.sum(weights)
    return {name: float(w) for name, w in zip(performance.keys(), weights)}


class ModelRegistryMixin:
    """Mixin to provide model registry methods for the predictor."""
    def create_model_registry(self, config, cold_start=False, history_days=0, num_outputs=11):
        return create_model_registry(config, cold_start, history_days, num_outputs)

    def compute_ensemble_weights(self, performance):
        return compute_ensemble_weights(performance)
