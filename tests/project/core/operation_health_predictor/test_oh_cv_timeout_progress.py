"""
Focused tests for the hardened Operation Health candidate-CV stage.

Covers:
- shared subprocess runner in "mae" mode: success, SIGKILL-on-timeout, error
- slow estimator timeout excludes candidate and run continues
- failed estimator excludes candidate and run continues
- all candidates failing/timeout raises a clear RuntimeError
- progress / per-model/fold timing recorded on predictor.cv_timing
- exactly one final full-data fit (the winner), after CV selection
- cv_folds default <=2 and cv_n_jobs default <=1
"""
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from core.common.cv_runner import evaluate_fold_in_subprocess
from core.nps_predictor.cv_test_helpers import BoomEstimator, SlowEstimator


def _oh_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, 5)).astype("float32"),
        columns=[f"f{i}" for i in range(5)],
    )
    y = pd.Series(rng.uniform(0.0, 100.0, n), dtype=np.float32)
    return X, y


def _fake_predictor(n_days=120):
    from core.operation_health_predictor.config import Config

    p = _FakePredictor()
    p.config = Config(
        n_estimators=5,
        mlp_hidden_layers=(4,),
        mlp_max_iter=2,
        use_cyclical_dates=False,
        clip_outliers=False,
    )
    p.config.cv_folds = 2
    p.config.cv_timeout = 5.0
    dates = pd.Series(pd.date_range("2026-01-01", periods=n_days, freq="D"))
    p._training_dates = dates.reset_index(drop=True)
    return p


def _ok_result(Xtr, ytr, Xva, yva):
    m = Ridge().fit(Xtr, ytr)
    pred = m.predict(Xva)
    return {
        "status": "ok",
        "score": float(mean_absolute_error(yva, pred)),
        "elapsed": 0.1,
    }


# ---------------------------------------------------------------------------
# Shared runner, OH ("mae") mode
# ---------------------------------------------------------------------------

def test_runner_mae_success():
    X, y = _oh_data()
    res = evaluate_fold_in_subprocess(
        "Ridge", Ridge(),
        X.iloc[:90], y[:90], X.iloc[90:], y[90:],
        timeout=60, metric="mae",
    )
    assert res["status"] == "ok"
    assert "score" in res
    assert isinstance(res["elapsed"], float)


def test_runner_mae_timeout_sigkills_slow_candidate():
    X, y = _oh_data()
    t0 = time.monotonic()
    res = evaluate_fold_in_subprocess(
        "Slow", SlowEstimator(delay=30),
        X.iloc[:90], y[:90], X.iloc[90:], y[90:],
        timeout=0.5, metric="mae",
    )
    wall = time.monotonic() - t0
    assert res["status"] == "timeout"
    assert wall < 10.0


def test_runner_mae_error():
    X, y = _oh_data()
    res = evaluate_fold_in_subprocess(
        "Boom", BoomEstimator(),
        X.iloc[:90], y[:90], X.iloc[90:], y[90:],
        timeout=30, metric="mae",
    )
    assert res["status"] == "error"


# ---------------------------------------------------------------------------
# Trainer behavior
# ---------------------------------------------------------------------------

def test_oh_timeout_excludes_candidate_and_continues(monkeypatch):
    import core.operation_health_predictor.trainer as trainer_mod

    X, y = _oh_data()
    predictor = _fake_predictor()
    predictor.create_model_registry = lambda cfg, cold_start=False: {
        "Good": Ridge(),
        "Slow": SlowEstimator(delay=30),
    }

    calls = []

    def fake(name, model, Xtr, ytr, Xva, yva, timeout, metric="mae", heartbeat=None):
        calls.append(name)
        if name == "Slow":
            return {"status": "timeout", "timeout": timeout, "elapsed": timeout}
        return _ok_result(Xtr, ytr, Xva, yva)

    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", fake)

    trainer_mod.TrainerMixin._rolling_origin_train(
        predictor, X, y, progress=None
    )

    assert set(calls) == {"Good", "Slow"}
    assert predictor.model_name == "Good"
    assert set(predictor.algorithm_performance) == {"Good"}
    assert "Slow" in predictor.cv_timing["timeouts"]


def test_oh_error_excludes_candidate_and_continues(monkeypatch):
    import core.operation_health_predictor.trainer as trainer_mod

    X, y = _oh_data()
    predictor = _fake_predictor()
    predictor.create_model_registry = lambda cfg, cold_start=False: {
        "Broken": Ridge(),
        "Good": Ridge(),
    }

    def fake(name, model, Xtr, ytr, Xva, yva, timeout, metric="mae", heartbeat=None):
        if name == "Broken":
            return {"status": "error", "error": "boom", "elapsed": 0.1}
        return _ok_result(Xtr, ytr, Xva, yva)

    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", fake)

    trainer_mod.TrainerMixin._rolling_origin_train(
        predictor, X, y, progress=None
    )

    assert predictor.model_name == "Good"
    assert set(predictor.algorithm_performance) == {"Good"}
    assert "Broken" in [n for n, _ in predictor.cv_timing["errors"]]


def test_oh_all_candidates_fail_raises_clear_error(monkeypatch):
    import core.operation_health_predictor.trainer as trainer_mod

    X, y = _oh_data()
    predictor = _fake_predictor()
    predictor.create_model_registry = lambda cfg, cold_start=False: {
        "A": Ridge(),
        "B": Ridge(),
    }

    def fake(name, model, Xtr, ytr, Xva, yva, timeout, metric="mae", heartbeat=None):
        return {"status": "timeout", "timeout": timeout, "elapsed": timeout}

    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", fake)

    with pytest.raises(RuntimeError) as excinfo:
        trainer_mod.TrainerMixin._rolling_origin_train(
            predictor, X, y, progress=None
        )
    assert "No model produced valid predictions" in str(excinfo.value)


def test_oh_progress_and_timing_recorded(monkeypatch):
    import core.operation_health_predictor.trainer as trainer_mod
    from core.common.training_progress import TrainingProgress

    X, y = _oh_data()
    predictor = _fake_predictor()
    predictor.create_model_registry = lambda cfg, cold_start=False: {
        "Good": Ridge(),
    }
    progress = TrainingProgress()

    def fake(name, model, Xtr, ytr, Xva, yva, timeout, metric="mae", heartbeat=None):
        for _ in range(3):
            if heartbeat is not None:
                heartbeat()
        return _ok_result(Xtr, ytr, Xva, yva)

    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", fake)

    trainer_mod.TrainerMixin._rolling_origin_train(
        predictor, X, y, progress=progress
    )

    snap = progress.snapshot()
    assert snap["total_models"] == 1
    assert predictor.cv_timing["completed"] == ["Good"]
    assert predictor.cv_timing["model_total_elapsed"]["Good"] >= 0.0
    assert predictor.cv_timing["fold_times"]


def test_oh_exactly_one_final_full_data_fit(tmp_path, monkeypatch):
    """train() performs exactly one full-data refit of the winner."""
    import core.operation_health_predictor.trainer as trainer_mod
    from core.operation_health_predictor.predictor import OperationalHealthPredictor
    from sklearn.base import clone as real_clone

    csv_path = tmp_path / "oh_train.csv"
    _write_oh_csv(csv_path, n_days=60)

    final_fit_calls = []

    class _FitSpy:
        def __init__(self, est, observer):
            self._est = est
            self._observer = observer

        def fit(self, X, y, **kwargs):
            self._observer(X, y)
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

    def spy_clone(estimator):
        return _FitSpy(real_clone(estimator), lambda X, y: final_fit_calls.append(len(X)))

    def fake(name, model, Xtr, ytr, Xva, yva, timeout, metric="mae", heartbeat=None):
        return _ok_result(Xtr, ytr, Xva, yva)

    monkeypatch.setattr(trainer_mod, "clone", spy_clone)
    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", fake)

    predictor = OperationalHealthPredictor()
    # Keep the test fast: skip SHAP / permutation importance / metadata.
    predictor._compute_shap = lambda X: None
    predictor._compute_feature_importance = lambda X, y: None
    predictor._update_metadata = lambda X, y: None

    predictor.train(str(csv_path))

    assert predictor.trained is True
    # Exactly one full-data fit happened, and it was on the full matrix.
    assert len(final_fit_calls) == 1
    assert set(predictor._all_models) == {predictor.model_name}
    assert predictor._all_models[predictor.model_name] is predictor.model
    assert predictor.cv_timing["completed"]


def test_oh_config_defaults_cheap():
    from core.operation_health_predictor.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.cv_folds <= 2
    assert DEFAULT_CONFIG.cv_n_jobs <= 1
    assert DEFAULT_CONFIG.cv_timeout >= 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_oh_csv(path, n_days=60, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    records = []
    for day in dates:
        records.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "target_quality": float(rng.uniform(80, 100)),
                "actual_quality": float(rng.uniform(80, 100)),
                "target_competency": float(rng.uniform(80, 100)),
                "actual_competency": float(rng.uniform(80, 100)),
                "target_attendance": float(rng.uniform(80, 100)),
                "actual_attendance": float(rng.uniform(80, 100)),
                "target_release_rate": float(rng.uniform(50, 90)),
                "actual_release_rate": float(rng.uniform(50, 90)),
                "target_transfer_rate": float(rng.uniform(0, 20)),
                "actual_transfer_rate": float(rng.uniform(0, 20)),
                "total_calls_received": int(rng.uniform(2000, 5000)),
                "operational_intelligence_factor": float(rng.uniform(-100, 100)),
                "operational_health": float(rng.uniform(0, 100)),
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


class _FakePredictor:
    def __init__(self):
        self.config = None
        self.model = None
        self.model_name = None
        self.algorithm_performance = {}
        self._all_models = {}
        self.cv_timing = {}
        self._training_dates = None
