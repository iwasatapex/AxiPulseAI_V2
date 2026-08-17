"""
AxiPulseAI – Full Model Registry (all models)
"""
import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
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

def create_model_registry(config, cold_start=False, history_days=0, num_outputs=1):
    v = 1 if config.verbose else 0

    # MLP pipeline
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
        "n_jobs": 1,
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

    # CatBoost
    if CatBoostRegressor is not None:
        safe_add("CatBoost", lambda: CatBoostRegressor(
            iterations=config.n_estimators,
            learning_rate=config.learning_rate,
            random_seed=config.random_state,
            verbose=config.verbose,
            thread_count=1,
        ))

    # LightGBM
    if LGBMRegressor is not None:
        safe_add("LightGBM", lambda: LGBMRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
            verbose=-1 if not v else v,
            n_jobs=1,
        ))

    # XGBoost
    if XGBRegressor is not None:
        safe_add("XGBoost", lambda: XGBRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
            verbosity=1 if v else 0,
            n_jobs=1,
        ))

    # ExtraTrees
    safe_add("ExtraTrees", lambda: ExtraTreesRegressor(**et_kwargs))
    # RandomForest
    safe_add("RandomForest", lambda: RandomForestRegressor(**rf_kwargs))
    # HistGradientBoosting
    safe_add("HistGradientBoosting", lambda: HistGradientBoostingRegressor(**hgb_kwargs))
    # GradientBoosting
    safe_add("GradientBoosting", lambda: GradientBoostingRegressor(**gb_kwargs))
    # MLP
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
    def create_model_registry(self, config, cold_start=False, history_days=0, num_outputs=1):
        return create_model_registry(config, cold_start, history_days, num_outputs)

    def compute_ensemble_weights(self, performance):
        return compute_ensemble_weights(performance)


def safe_add(*args, **kwargs):
    """Compatibility helper matching the registry's safe-add semantics.

    Supported forms:

        safe_add(models, name, model_factory, verbose=False)
        safe_add(name, model_factory, models, verbose=False)
        safe_add(name=name, model_factory=factory, models=models, verbose=False)

    The helper returns the supplied models mapping.
    """
    models = kwargs.get("models")
    name = kwargs.get("name")
    model_factory = kwargs.get("model_factory")
    verbose = bool(kwargs.get("verbose", False))

    if args:
        if len(args) >= 3 and isinstance(args[0], dict):
            models = args[0]
            name = args[1]
            model_factory = args[2]
            if len(args) >= 4:
                verbose = bool(args[3])
        elif len(args) >= 3:
            name = args[0]
            model_factory = args[1]
            models = args[2]
            if len(args) >= 4:
                verbose = bool(args[3])

    if models is None or name is None or model_factory is None:
        raise TypeError(
            "safe_add() requires models, name and model_factory."
        )

    try:
        models[name] = model_factory()
    except Exception as exc:
        if verbose:
            print(f"Skipping {name}: {exc}")

    return models
