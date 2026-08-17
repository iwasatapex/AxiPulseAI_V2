"""
Tests for the RAM-safety hardening of NPS candidate CV and the final refit.

Guarantees covered:
- Bounded selection sample never exceeds the configured size.
- A CV worker never receives the full X/y matrix (only bounded fold slices).
- MLP keeps its stricter CV timeout and is excluded (with an explicit reason)
  when it exceeds the per-fold resource limit.
- Exactly one full-data final fit happens.
- No candidate models are retained before the final full-data refit.
- The GPU path is unchanged (eligible families only; MLP/forests stay CPU).
- The CV subprocess reports peak RSS (smoke test) so a fix can be verified.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import core.nps_predictor.trainer as trainer_mod
from core.nps_predictor.config import DEFAULT_CONFIG
from core.nps_predictor.trainer import _select_temporal_sample


# --------------------------------------------------------------------------- #
# 1. Bounded CV sample never exceeds configured size
# --------------------------------------------------------------------------- #
def test_bounded_sample_never_exceeds_configured_size():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(20000, 5)).astype("float32"),
        columns=[f"f{i}" for i in range(5)],
    )
    y = np.abs(rng.normal(5.0, 2.0, size=(20000, 11))).astype("float32")
    dates = pd.Series(pd.date_range("2026-01-01", periods=20000, freq="h"))

    Xs, ys, ds = _select_temporal_sample(X, y, dates, 500, 42)

    assert len(Xs) <= 500
    assert len(ys) == len(Xs)
    assert len(ds) == len(Xs)
    # The sample is a strict subset of the full matrix.
    assert len(Xs) < len(X)
    # Chronological order is preserved.
    assert ds.is_monotonic_increasing


def test_default_selection_sample_is_500_rows():
    assert DEFAULT_CONFIG.sample_size <= 500


def test_default_cv_is_serial_two_folds():
    assert DEFAULT_CONFIG.cv_folds <= 2
    assert DEFAULT_CONFIG.cv_n_jobs == 1
    assert DEFAULT_CONFIG.cv_memory_ceiling_mb == 2048.0


# --------------------------------------------------------------------------- #
# 2. Worker never receives the full X/y
# --------------------------------------------------------------------------- #
def _cv_frame(n=100, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)).astype("float32"),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = np.abs(rng.normal(5.0, 2.0, size=(n, 11))).astype("float32")
    dates = pd.Series(pd.date_range("2026-01-01", periods=n, freq="D"))
    return X, y, dates


def _ok_result(*_a, **_k):
    return {
        "status": "ok",
        "nps_mae": 0.5,
        "bucket_mae": 0.4,
        "elapsed": 0.1,
        "peak_rss_mb": 100.0,
        "worker_pid": 12345,
    }


def test_worker_never_receives_full_xy(monkeypatch):
    X, y, dates = _cv_frame(n=100)
    seen = []

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout,
                        heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        seen.append((len(Xtr), len(ytr), len(Xva), len(yva)))
        return _ok_result()

    monkeypatch.setattr(trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess)
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: {"Ridge": Ridge()},
    )

    predictor = _FakePredictor()
    from core.nps_predictor.config import Config

    predictor.config = Config(
        n_estimators=2,
        cv_folds=2,
        cv_n_jobs=1,
        cv_memory_ceiling_mb=2048.0,
        cv_mlp_timeout=5.0,
    )

    trainer_mod.rolling_origin_train(predictor, X, y, dates=dates, progress=None)

    assert seen, "no fold was dispatched to a worker"
    for ntr, nyr, nva, nyva in seen:
        # Each worker receives only a bounded fold slice, never the full matrix.
        assert ntr < len(X)
        assert nva < len(X)
        assert (ntr + nva) <= len(X)


# --------------------------------------------------------------------------- #
# 3. MLP stricter timeout + resource exclusion
# --------------------------------------------------------------------------- #
def test_mlp_uses_stricter_timeout(monkeypatch):
    X, y, dates = _cv_frame(n=100)
    timeouts = {}

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout,
                        heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        timeouts[name] = timeout
        return _ok_result()

    monkeypatch.setattr(trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess)
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: {
            "MLP": Ridge(),
            "Ridge": Ridge(),
        },
    )

    predictor = _FakePredictor()
    from core.nps_predictor.config import Config

    predictor.config = Config(
        n_estimators=2,
        cv_folds=2,
        cv_n_jobs=1,
        cv_timeout=60.0,
        cv_mlp_timeout=5.0,
        cv_memory_ceiling_mb=2048.0,
    )

    trainer_mod.rolling_origin_train(predictor, X, y, dates=dates, progress=None)

    assert timeouts["MLP"] == 5.0  # stricter timeout for MLP
    assert timeouts["Ridge"] == 60.0  # default timeout for others


def test_mlp_excluded_on_resource_limit_with_reason(monkeypatch):
    X, y, dates = _cv_frame(n=100)

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout,
                        heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        if name == "MLP":
            return {
                "status": "memory_limit",
                "ceiling_mb": 2048.0,
                "peak_rss_mb": 3000.0,
                "elapsed": 2.0,
            }
        return _ok_result()

    monkeypatch.setattr(trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess)
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: {
            "MLP": Ridge(),
            "Ridge": Ridge(),
        },
    )

    predictor = _FakePredictor()
    from core.nps_predictor.config import Config

    predictor.config = Config(
        n_estimators=2,
        cv_folds=2,
        cv_n_jobs=1,
        cv_mlp_timeout=5.0,
        cv_memory_ceiling_mb=2048.0,
    )

    trainer_mod.rolling_origin_train(predictor, X, y, dates=dates, progress=None)

    # MLP is excluded with an explicit resource-limit reason.
    assert any(n == "MLP" for n, _ in predictor.cv_timing["memory_limits"])
    assert "MLP" not in predictor.algorithm_performance
    assert predictor.model_name == "Ridge"


# --------------------------------------------------------------------------- #
# 4/5. Exactly one full-data final fit; no candidates retained before it
# --------------------------------------------------------------------------- #
class _CountingRidge(Ridge):
    fit_count = 0
    all_models_at_fit = None

    def fit(self, X, y, **kwargs):
        type(self).fit_count += 1
        type(self).all_models_at_fit = dict(getattr(self, "_pred_ref", {})._all_models)
        return super().fit(X, y, **kwargs)


def _write_synthetic_csv(tmp_path, n_rows=400, n_dates=40):
    rng = np.random.default_rng(0)
    dates = (
        pd.Timestamp("2024-01-01")
        + pd.to_timedelta(rng.integers(0, n_dates, size=n_rows), unit="D")
    )
    total_calls = rng.integers(200, 2000, size=n_rows).astype(np.float64)
    raw = rng.dirichlet(np.ones(11), size=n_rows)
    scores = np.round(raw * total_calls[:, None]).astype(np.float64)
    total_surveys = scores.sum(axis=1).astype(np.float64)

    df = pd.DataFrame(
        {
            "date": dates,
            "operational_health": rng.uniform(0, 120, size=n_rows),
            "business_intelligence_factor": rng.uniform(0, 100, size=n_rows),
            "member_intelligence_factor": rng.uniform(0, 100, size=n_rows),
            "target_release_rate": rng.uniform(0, 100, size=n_rows),
            "actual_release_rate": rng.uniform(0, 100, size=n_rows),
            "total_calls_received": total_calls,
            "total_surveys": total_surveys,
        }
    )
    for i in range(11):
        df[f"score_{i}"] = scores[:, i]
    df["promoters"] = df["score_9"] + df["score_10"]
    df["passives"] = df["score_7"] + df["score_8"]
    df["detractors"] = (
        df["score_0"] + df["score_1"] + df["score_2"] + df["score_3"]
        + df["score_4"] + df["score_5"] + df["score_6"]
    )

    path = tmp_path / "nps_data.csv"
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(autouse=True)
def _reset_counter():
    _CountingRidge.fit_count = 0
    _CountingRidge.all_models_at_fit = None
    yield


def test_single_final_fit_and_no_candidates_retained(tmp_path, monkeypatch):
    from core.nps_predictor.config import Config
    from core.nps_predictor.predictor import NPSPredictor

    path = _write_synthetic_csv(tmp_path)

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout,
                        heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        return _ok_result()

    monkeypatch.setattr(trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess)
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: {
            "Ridge": _CountingRidge(),
        },
    )

    predictor = NPSPredictor(
        config=Config(
            n_estimators=2,
            cv_folds=2,
            cv_n_jobs=1,
            cv_timeout=10.0,
            cv_mlp_timeout=5.0,
            cv_memory_ceiling_mb=2048.0,
            use_gpu=False,
            enable_shap=False,
            sample_for_selection=True,
            sample_size=500,
        )
    )
    _CountingRidge._pred_ref = predictor

    predictor.train(path)

    # Exactly one full-data final fit (CV was subprocess-monkeypatched, no fit).
    assert _CountingRidge.fit_count == 1
    # No candidate models retained at the moment of the final refit.
    assert _CountingRidge.all_models_at_fit == {}
    # Only the selected (full-data-fitted) model is retained after training.
    assert set(predictor._all_models) == {predictor.model_name}
    assert predictor.trained is True


# --------------------------------------------------------------------------- #
# 6. GPU path unchanged
# --------------------------------------------------------------------------- #
def test_gpu_path_unchanged_mlp_forests_stay_cpu(monkeypatch):
    from core.nps_predictor import gpu as gpu_mod
    from core.nps_predictor.config import Config

    monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
    monkeypatch.setattr(gpu_mod, "lightgbm_gpu_supported", lambda: True)
    gpu_mod.reset_cache()

    cfg = Config(use_gpu=True)

    # GPU-eligible families can be selected for GPU.
    assert gpu_mod.select_final_fit_device("CatBoost", cfg) == "gpu"
    assert gpu_mod.select_final_fit_device("XGBoost", cfg) == "gpu"

    # MLP and forests always stay on CPU.
    assert gpu_mod.select_final_fit_device("MLP", cfg) == "cpu"
    assert gpu_mod.select_final_fit_device("RandomForest", cfg) == "cpu"
    assert gpu_mod.select_final_fit_device("GradientBoosting", cfg) == "cpu"

    # GPU disabled -> CPU for everyone.
    cfg_no_gpu = Config(use_gpu=False)
    assert gpu_mod.select_final_fit_device("CatBoost", cfg_no_gpu) == "cpu"

    gpu_mod.reset_cache()


def test_apply_gpu_params_returns_false_for_cpu_only(monkeypatch):
    from core.nps_predictor import gpu as gpu_mod
    from core.nps_predictor.config import Config

    monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
    gpu_mod.reset_cache()

    cfg = Config(use_gpu=True)
    assert gpu_mod.apply_gpu_params(object(), "MLP", cfg) is False
    assert gpu_mod.apply_gpu_params(object(), "RandomForest", cfg) is False

    gpu_mod.reset_cache()


# --------------------------------------------------------------------------- #
# 7. Smoke test: CV worker reports peak RSS
# --------------------------------------------------------------------------- #
def test_smoke_cv_worker_records_peak_rss():
    """A real fold subprocess must report a finite peak RSS (fix verifiable)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(40, 5)).astype("float32"),
        columns=[f"f{i}" for i in range(5)],
    )
    y = np.abs(rng.normal(5.0, 2.0, size=(40, 11))).astype("float32")

    res = trainer_mod._evaluate_fold_in_subprocess(
        "Ridge",
        Ridge(),
        X.iloc[:30],
        y[:30],
        X.iloc[30:],
        y[30:],
        timeout=60,
    )

    assert res["status"] == "ok"
    peak = res.get("peak_rss_mb")
    assert peak is not None
    assert peak == peak  # not NaN
    assert peak > 0
    assert res.get("worker_pid") is not None


class _FakePredictor:
    def __init__(self):
        self.config = None
        self._all_models = {}
        self.model = None
        self.model_name = None
        self.algorithm_performance = {}
        self.algorithm_bucket_mae = {}
        self.cv_timing = {}
        self.training_rows = 0
        self.history_days = 0
        self._feature_importance = {}
        self._feature_stats = {}
