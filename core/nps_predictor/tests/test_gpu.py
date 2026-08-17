"""
Tests for optional GPU acceleration of the FINAL NPS training fit.

Guarantees covered:
- GPU disabled -> CPU.
- GPU requested but unavailable -> CPU fallback.
- CatBoost selected -> GPU configuration.
- XGBoost selected -> GPU configuration.
- CPU-only models (MLP / forests) -> CPU.
- No GPU use during CV / model selection (registry stays CPU).
- The selected model is fit exactly once on the complete dataset.
- Persistence (save/load/predict) still works.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone as _real_clone

from core.nps_predictor import Config, NPSPredictor
from core.nps_predictor import gpu
from core.nps_predictor.models import create_model_registry


def test_gpu_disabled_uses_cpu(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu_available_cache", True)
    config = Config(use_gpu=False)
    assert gpu.select_final_fit_device("CatBoost", config) == "cpu"
    assert gpu.select_final_fit_device("XGBoost", config) == "cpu"


def test_gpu_requested_but_unavailable_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu_available_cache", False)
    config = Config(use_gpu=True)
    assert gpu.select_final_fit_device("CatBoost", config) == "cpu"
    assert gpu.select_final_fit_device("XGBoost", config) == "cpu"


def test_env_override_forces_cpu(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu_available_cache", None)
    monkeypatch.setenv("AXIPULSE_DISABLE_GPU", "1")
    assert gpu.gpu_available() is False
    config = Config(use_gpu=True)
    assert gpu.select_final_fit_device("CatBoost", config) == "cpu"


def test_catboost_selected_uses_gpu_configuration(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu_available_cache", True)
    config = Config(use_gpu=True, use_catboost_multi=True)
    model = create_model_registry(config)["CatBoost"]
    assert gpu.select_final_fit_device("CatBoost", config) == "gpu"
    assert gpu.apply_gpu_params(model, "CatBoost", config) is True
    assert model.get_params().get("task_type") == "GPU"
    assert model.get_params().get("devices") == "0"


def test_catboost_wrapped_uses_gpu_configuration(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu_available_cache", True)
    config = Config(use_gpu=True, use_catboost_multi=False)
    model = create_model_registry(config)["CatBoost"]
    assert isinstance(model, MultiOutputRegressor)
    assert gpu.apply_gpu_params(model, "CatBoost", config) is True
    est = model.estimator
    assert est.get_params().get("task_type") == "GPU"
    assert est.get_params().get("devices") == "0"


def test_xgboost_selected_uses_gpu_configuration(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu_available_cache", True)
    config = Config(use_gpu=True)
    model = create_model_registry(config)["XGBoost"]
    assert gpu.select_final_fit_device("XGBoost", config) == "gpu"
    assert gpu.apply_gpu_params(model, "XGBoost", config) is True
    est = model.estimator
    assert est.get_params().get("tree_method") == "hist"
    assert est.get_params().get("device") == "cuda"
    assert est.get_params().get("n_jobs") == 1


def test_lightgbm_without_gpu_build_stays_cpu(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu_available_cache", True)
    monkeypatch.setattr(gpu, "_lgb_gpu_cache", False)
    config = Config(use_gpu=True)
    assert gpu.select_final_fit_device("LightGBM", config) == "cpu"


def test_cpu_only_models_stay_on_cpu(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu_available_cache", True)
    config = Config(use_gpu=True)
    for name in (
        "MLP",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "GradientBoosting",
    ):
        assert gpu.select_final_fit_device(name, config) == "cpu"
        model = create_model_registry(config)[name]
        assert gpu.apply_gpu_params(model, name, config) is False


def test_no_gpu_used_during_cv_registry_stays_cpu(monkeypatch):
    """CV uses the CPU registry; no candidate is ever GPU-configured."""
    monkeypatch.setattr(gpu, "_gpu_available_cache", True)
    config = Config(use_gpu=True, use_catboost_multi=True)
    registry = create_model_registry(config)
    cat = registry["CatBoost"]
    assert cat.get_params().get("task_type") != "GPU"
    xgb = registry["XGBoost"]
    assert xgb.estimator.get_params().get("device") != "cuda"


def _write_training_csv(path, n_days=90, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    records = []
    for day in dates:
        scores = rng.integers(0, 40, size=11)
        records.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "operational_health": float(rng.uniform(40, 100)),
                "business_intelligence_factor": float(rng.uniform(-100, 100)),
                "member_intelligence_factor": float(rng.uniform(-100, 100)),
                "target_release_rate": float(rng.uniform(0, 100)),
                "actual_release_rate": float(rng.uniform(0, 100)),
                "total_calls_received": int(rng.integers(100, 2000)),
                "total_surveys": int(scores.sum()),
                "survey_rate": float(rng.uniform(0.2, 1.0)),
                "target_quality": float(rng.uniform(0, 100)),
                "quality": float(rng.uniform(0, 100)),
                "target_competency": float(rng.uniform(0, 100)),
                "competency": float(rng.uniform(0, 100)),
                "target_attendance": float(rng.uniform(0, 100)),
                "attendance": float(rng.uniform(0, 100)),
                "target_transfer": float(rng.uniform(0, 100)),
                "transfer": float(rng.uniform(0, 100)),
                "promoters": int(scores[9:].sum()),
                "passives": int(scores[7:9].sum()),
                "detractors": int(scores[:7].sum()),
                **{f"score_{i}": int(scores[i]) for i in range(11)},
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


def _small_config(**overrides):
    params = dict(
        n_estimators=5,
        mlp_hidden_layers=(4,),
        mlp_max_iter=2,
        use_cyclical_dates=False,
        clip_outliers=False,
        sample_for_selection=True,
        sample_size=40,
        history_buffer_days=3,
        cv_n_jobs=1,
        use_gpu=False,
    )
    params.update(overrides)
    return Config(**params)


class _FitSpy:
    def __init__(self, estimator, observer):
        self._est = estimator
        self._observer = observer

    def fit(self, X, y, **kwargs):
        self._observer(X)
        return self._est.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self._est.predict(X, **kwargs)

    def get_params(self, deep=True):
        return self._est.get_params(deep)

    def set_params(self, **params):
        self._est.set_params(**params)
        return self

    def __getattr__(self, name):
        return getattr(self._est, name)


def test_only_one_final_full_data_fit(tmp_path, monkeypatch):
    """The selected model is fit exactly once on the complete dataset."""
    import core.nps_predictor.trainer as trainer_mod

    n_days = 90
    full_rows = n_days - 1

    config = _small_config()
    predictor = NPSPredictor(config=config)

    csv_path = tmp_path / "nps_train.csv"
    _write_training_csv(csv_path, n_days=n_days)

    full_size_fits = []

    def observe(X):
        if len(X) == full_rows:
            full_size_fits.append(len(X))

    def spy_clone(estimator):
        return _FitSpy(_real_clone(estimator), observe)

    monkeypatch.setattr(trainer_mod, "clone", spy_clone)

    predictor.train(str(csv_path))

    assert predictor.trained is True
    assert len(full_size_fits) == 1
    assert full_size_fits == [full_rows]


def test_persistence_still_works(tmp_path):
    """GPU/CPU training save->load->predict round trip."""
    config = _small_config()
    csv_path = tmp_path / "nps_train.csv"
    model_path = tmp_path / "nps_model.pkl"
    _write_training_csv(csv_path, n_days=90)

    trainer_predictor = NPSPredictor(config=config)
    trainer_predictor.train(str(csv_path))
    assert trainer_predictor.trained is True
    trainer_predictor.save_model(str(model_path))

    loaded = NPSPredictor(config=config)
    loaded.load_model(str(model_path))
    assert loaded.trained is True
    assert loaded.model is not None

    row = {
        "date": "2026-04-02",
        "operational_health": 90.0,
        "business_intelligence_factor": 10.0,
        "member_intelligence_factor": 10.0,
        "target_release_rate": 80.0,
        "actual_release_rate": 75.0,
        "total_calls_received": 2000,
        "total_surveys": 200,
        "survey_rate": 0.1,
        "target_quality": 80.0,
        "quality": 78.0,
        "target_competency": 85.0,
        "competency": 82.0,
        "target_attendance": 90.0,
        "attendance": 88.0,
        "target_transfer": 60.0,
        "transfer": 58.0,
    }
    result = loaded.predict(row)
    assert isinstance(result, dict)
    assert "nps" in result
    assert -100.0 <= result["nps"] <= 100.0
