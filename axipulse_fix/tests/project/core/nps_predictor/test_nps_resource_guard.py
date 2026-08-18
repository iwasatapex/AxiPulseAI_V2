"""
Focused tests for NPS final-fit resource safety.

Covers:
- ExtraTrees / RandomForest final-fit n_jobs=1 (outer + inner, no nested
  parallelism).
- Final-fit memory guard: raises a clear diagnostic when the selected tree
  ensemble cannot fit under the budget; reduces parallelism first; optionally
  downscales the estimator count.
- LightGBM / XGBoost 11-output fits work through MultiOutputRegressor.
- XGBoost subprocess CV smoke test (the eval_set path that previously failed).
- GPU CatBoost final fit vs CPU-only ExtraTrees final fit.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from core.nps_predictor import resource_guard as rg
from core.nps_predictor.config import Config
from core.nps_predictor.models import create_model_registry


def _tree_model(name, n_estimators=500, n_jobs=4):
    cfg = Config(
        n_estimators=n_estimators,
        random_state=42,
    )
    registry = create_model_registry(cfg, num_outputs=11)
    return registry[name]


class _FakePredictor:
    def __init__(self, model, model_name, **cfg_kwargs):
        self.config = Config(**cfg_kwargs)
        self.model = model
        self.model_name = model_name


def _xy(rows=100_620, cols=34, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(rows, cols)).astype("float32"),
        columns=[f"f{i}" for i in range(cols)],
    )
    y = np.abs(rng.normal(5.0, 2.0, size=(rows, 11))).astype("float32")
    return X, y


# --------------------------------------------------------------------------- #
# 1. ExtraTrees / RandomForest final-fit n_jobs=1, no nested parallelism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["ExtraTrees", "RandomForest"])
def test_cpu_tree_final_fit_uses_n_jobs_1(name):
    model = _tree_model(name, n_estimators=50)
    assert isinstance(model, MultiOutputRegressor)
    # Simulate the registry's default parallel inner estimator.
    model.estimator.set_params(n_jobs=4)
    model.set_params(n_jobs=2)

    cfg = Config(final_cpu_n_jobs=1)
    effective = rg.apply_final_cpu_config(model, name, cfg)

    assert effective == 1
    # Outer MultiOutputRegressor n_jobs == 1.
    assert model.n_jobs == 1
    # Inner estimator n_jobs == 1 -> no nested parallelism.
    assert model.estimator.n_jobs == 1


def test_no_nested_parallelism_et():
    model = _tree_model("ExtraTrees", n_estimators=30)
    model.estimator.set_params(n_jobs=4)
    model.set_params(n_jobs=2)
    rg.apply_final_cpu_config(model, "ExtraTrees", Config(final_cpu_n_jobs=1))
    assert model.n_jobs == 1
    assert model.estimator.n_jobs == 1


# --------------------------------------------------------------------------- #
# 2. Final-fit memory guard
# --------------------------------------------------------------------------- #
def test_tree_memory_estimate_is_large_at_100k():
    est = ExtraTreesRegressor(n_estimators=500)
    est_mb = rg.estimate_tree_fit_memory_mb("ExtraTrees", est, 100_620, 11, n_jobs=1)
    assert est_mb is not None
    # 500 trees x 11 outputs on ~100k rows is inherently huge.
    assert est_mb > 10_000  # > 10 GB


def test_memory_guard_raises_when_over_budget():
    model = _tree_model("ExtraTrees", n_estimators=500)
    predictor = _FakePredictor(model, "ExtraTrees", final_fit_memory_budget_mb=1024.0)
    X, y = _xy()
    with pytest.raises(RuntimeError) as ei:
        rg.guard_final_fit(predictor, X, y)
    msg = str(ei.value)
    assert "ExtraTrees" in msg
    assert "final_fit_memory_budget_mb" in msg
    # No rows were dropped / model not substituted: model is untouched here.
    assert predictor.model is model


def test_memory_guard_passes_within_budget_and_sets_n_jobs_1():
    model = _tree_model("ExtraTrees", n_estimators=10)
    model.estimator.set_params(n_jobs=4)
    model.set_params(n_jobs=2)
    # Tiny rows => tiny trees, generous budget => no raise.
    predictor = _FakePredictor(
        model, "ExtraTrees", final_fit_memory_budget_mb=4096.0
    )
    X, y = _xy(rows=200)
    rg.guard_final_fit(predictor, X, y)  # must not raise
    assert model.n_jobs == 1
    assert model.estimator.n_jobs == 1


def test_memory_guard_downscales_when_enabled():
    model = _tree_model("ExtraTrees", n_estimators=500)
    predictor = _FakePredictor(
        model,
        "ExtraTrees",
        final_fit_memory_budget_mb=1024.0,
        final_fit_auto_downscale=True,
    )
    X, y = _xy(rows=5000)
    rg.guard_final_fit(predictor, X, y)  # must not raise
    assert 0 < model.estimator.n_estimators < 500


def test_memory_guard_raises_when_downscale_cannot_fit():
    model = _tree_model("ExtraTrees", n_estimators=500)
    predictor = _FakePredictor(
        model,
        "ExtraTrees",
        final_fit_memory_budget_mb=0.001,
        final_fit_auto_downscale=True,
    )
    X, y = _xy(rows=100_620)
    with pytest.raises(RuntimeError):
        rg.guard_final_fit(predictor, X, y)


def test_memory_guard_does_not_reject_gpu_fit():
    # A GPU final fit computes on VRAM, not system RAM; the hard-fail RAM guard
    # must not reject a GPU-eligible model even when the tree estimate is huge.
    model = _tree_model("ExtraTrees", n_estimators=500)
    predictor = _FakePredictor(
        model, "ExtraTrees", final_fit_memory_budget_mb=1.0
    )
    X, y = _xy(rows=100_620)
    rg.guard_final_fit(predictor, X, y, device="gpu")  # must not raise
    # The serial n_jobs=1 CPU config is still applied in the GPU path.
    assert model.n_jobs == 1
    assert model.estimator.n_jobs == 1


# --------------------------------------------------------------------------- #
# 3. LightGBM / XGBoost 11-output fits (the CV failures)
# --------------------------------------------------------------------------- #
def test_lightgbm_11_output_fit_through_multioutput():
    lgbm = pytest.importorskip("lightgbm")
    from lightgbm import LGBMRegressor

    model = MultiOutputRegressor(LGBMRegressor(n_estimators=5, verbose=-1, n_jobs=1))
    X, y = _xy(rows=120)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (120, 11)


def test_xgboost_11_output_fit_through_multioutput():
    xgb = pytest.importorskip("xgboost")
    from xgboost import XGBRegressor

    model = MultiOutputRegressor(XGBRegressor(n_estimators=5, verbosity=0, n_jobs=1))
    X, y = _xy(rows=120)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (120, 11)


def test_xgboost_subprocess_cv_smoke():
    """The eval_set-through-MultiOutputRegressor path that previously failed."""
    import core.nps_predictor.trainer as trainer_mod
    from xgboost import XGBRegressor

    X, y = _xy(rows=80)
    model = MultiOutputRegressor(XGBRegressor(n_estimators=5, verbosity=0, n_jobs=1))
    res = trainer_mod._evaluate_fold_in_subprocess(
        "XGBoost",
        model,
        X.iloc[:60],
        y[:60],
        X.iloc[60:],
        y[60:],
        timeout=60,
    )
    assert res["status"] == "ok"
    assert "nps_mae" in res
    assert res["peak_rss_mb"] == res["peak_rss_mb"]  # not NaN


def test_lightgbm_subprocess_cv_smoke():
    import core.nps_predictor.trainer as trainer_mod
    from lightgbm import LGBMRegressor

    X, y = _xy(rows=80)
    model = MultiOutputRegressor(LGBMRegressor(n_estimators=5, verbose=-1, n_jobs=1))
    res = trainer_mod._evaluate_fold_in_subprocess(
        "LightGBM",
        model,
        X.iloc[:60],
        y[:60],
        X.iloc[60:],
        y[60:],
        timeout=60,
    )
    assert res["status"] == "ok"
    assert "nps_mae" in res


# --------------------------------------------------------------------------- #
# 4. GPU / CPU device decisions
# --------------------------------------------------------------------------- #
def test_gpu_catboost_final_fit(monkeypatch):
    from core.nps_predictor import gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
    monkeypatch.setattr(gpu_mod, "lightgbm_gpu_supported", lambda: True)
    gpu_mod.reset_cache()

    cfg = Config(use_gpu=True)
    assert gpu_mod.select_final_fit_device("CatBoost", cfg) == "gpu"

    gpu_mod.reset_cache()


def test_cpu_only_extratrees_final_fit(monkeypatch):
    from core.nps_predictor import gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
    gpu_mod.reset_cache()

    cfg = Config(use_gpu=True)
    # GPU exists, but ExtraTrees is CPU-only and must never be claimed GPU.
    assert gpu_mod.select_final_fit_device("ExtraTrees", cfg) == "cpu"
    assert gpu_mod.select_final_fit_device("RandomForest", cfg) == "cpu"

    gpu_mod.reset_cache()
