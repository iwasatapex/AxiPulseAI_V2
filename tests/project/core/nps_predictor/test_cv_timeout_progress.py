"""
Focused tests for the hardened NPS candidate-CV stage:

- Per-model/per-fold progress logging (START / DONE / SCORE) with elapsed time.
- Configurable per-fold CV timeout via subprocess isolation.
- Timeout terminates/skips the candidate and continues with the rest.
- A failed/timed-out candidate is excluded from selection.
- If ALL candidates fail/timeout, training raises a clear error.
- No fake percentage: progress percent stays None for indeterminate stages and
  is only ever computed from real fold/model counts.
- CV heartbeat keeps the GUI state from looking frozen.
- Selection sample stays <=1000 rows and folds <=2.
"""
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from core.common.training_progress import TrainingProgress
from core.nps_predictor.cv_test_helpers import BoomEstimator, SlowEstimator
from core.nps_predictor.metrics import compute_nps_error
from core.nps_predictor.trainer import (
    _build_date_aware_splits,
    _evaluate_fold_in_subprocess,
)


def _make_cv_data(n=120, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)).astype("float32"),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = np.abs(rng.normal(5.0, 2.0, size=(n, 11))).astype("float32")
    return X, y


def _fake_predictor():
    from core.nps_predictor.config import Config

    predictor = _FakePredictor()
    predictor.config = Config(
        n_estimators=5,
        mlp_hidden_layers=(4,),
        mlp_max_iter=2,
        use_cyclical_dates=False,
        clip_outliers=False,
        history_buffer_days=3,
        cv_n_jobs=1,
        use_gpu=False,
    )
    return predictor


# ---------------------------------------------------------------------------
# Subprocess fold evaluation (real worker)
# ---------------------------------------------------------------------------

def test_fold_subprocess_returns_score():
    X, y = _make_cv_data()
    res = _evaluate_fold_in_subprocess(
        "Ridge",
        Ridge(),
        X.iloc[:90], y[:90],
        X.iloc[90:], y[90:],
        timeout=60,
    )
    assert res["status"] == "ok"
    assert "nps_mae" in res
    assert "bucket_mae" in res
    assert isinstance(res["elapsed"], float)


def test_fold_subprocess_timeout_sigkills_slow_candidate():
    """A slow fold is SIGKILLed promptly instead of hanging the run."""
    X, y = _make_cv_data()
    t0 = time.monotonic()
    res = _evaluate_fold_in_subprocess(
        "Slow",
        SlowEstimator(delay=30),
        X.iloc[:90], y[:90],
        X.iloc[90:], y[90:],
        timeout=0.5,
    )
    wall = time.monotonic() - t0
    assert res["status"] == "timeout"
    assert wall < 10.0  # must not hang the whole run


def test_fold_subprocess_error_surface():
    X, y = _make_cv_data()

    res = _evaluate_fold_in_subprocess(
        "Boom",
        BoomEstimator(),
        X.iloc[:90], y[:90],
        X.iloc[90:], y[90:],
        timeout=30,
    )
    assert res["status"] == "error"


# ---------------------------------------------------------------------------
# Timeout drives the trainer's exclusion / continue behavior
# ---------------------------------------------------------------------------

def _ok_result(Xtr, ytr, Xva, yva):
    m = Ridge().fit(Xtr, ytr)
    pred = m.predict(Xva)
    return {
        "status": "ok",
        "nps_mae": float(compute_nps_error(yva, pred)),
        "bucket_mae": float(mean_absolute_error(yva, pred)),
        "elapsed": 0.1,
    }


def test_timeout_excludes_candidate_and_continues(monkeypatch):
    """A timed-out candidate is excluded but the run still selects a winner."""
    import core.nps_predictor.trainer as trainer_mod

    X, y = _make_cv_data()
    dates = pd.Series(pd.date_range("2026-01-01", periods=len(X), freq="D"))
    splits = list(_build_date_aware_splits(dates, n_splits=2))

    base_models = {"Good": Ridge(), "Slow": SlowEstimator(delay=30)}

    calls = []

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout, heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        calls.append(name)
        if name == "Slow":
            return {"status": "timeout", "timeout": timeout, "elapsed": timeout}
        return _ok_result(Xtr, ytr, Xva, yva)

    monkeypatch.setattr(
        trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess
    )
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: base_models,
    )

    predictor = _fake_predictor()
    predictor.config.cv_timeout = 5.0
    predictor.config.cv_folds = 2

    trainer_mod.rolling_origin_train(predictor, X, y, dates=dates, progress=None)

    assert set(calls) == {"Good", "Slow"}
    # Slow candidate is excluded; a real winner is selected.
    assert predictor.model_name == "Good"
    assert set(predictor.algorithm_performance) == {"Good"}
    assert "Slow" in predictor.cv_timing["timeouts"]


def test_failed_candidate_excluded_but_run_continues(monkeypatch):
    import core.nps_predictor.trainer as trainer_mod

    X, y = _make_cv_data()
    dates = pd.Series(pd.date_range("2026-01-01", periods=len(X), freq="D"))

    base_models = {"Broken": Ridge(), "Good": Ridge()}

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout, heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        if name == "Broken":
            return {"status": "error", "error": "boom", "elapsed": 0.1}
        return _ok_result(Xtr, ytr, Xva, yva)

    monkeypatch.setattr(
        trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess
    )
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: base_models,
    )

    predictor = _fake_predictor()
    predictor.config.cv_timeout = 5.0
    predictor.config.cv_folds = 2

    trainer_mod.rolling_origin_train(predictor, X, y, dates=dates, progress=None)

    assert predictor.model_name == "Good"
    assert set(predictor.algorithm_performance) == {"Good"}
    assert "Broken" in [n for n, _ in predictor.cv_timing["errors"]]


def test_all_candidates_fail_raises_clear_error(monkeypatch):
    import core.nps_predictor.trainer as trainer_mod

    X, y = _make_cv_data()
    dates = pd.Series(pd.date_range("2026-01-01", periods=len(X), freq="D"))

    base_models = {"A": Ridge(), "B": Ridge()}

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout, heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        return {"status": "timeout", "timeout": timeout, "elapsed": timeout}

    monkeypatch.setattr(
        trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess
    )
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: base_models,
    )

    predictor = _fake_predictor()
    predictor.config.cv_timeout = 5.0
    predictor.config.cv_folds = 2

    with pytest.raises(RuntimeError) as excinfo:
        trainer_mod.rolling_origin_train(predictor, X, y, dates=dates, progress=None)
    assert "No model produced valid predictions" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Progress / heartbeat
# ---------------------------------------------------------------------------

def test_cv_progress_never_fakes_percentage():
    p = TrainingProgress()
    p.set_models(8)
    assert p.percent is None  # no fake % before any real unit completes
    p.start_fold(1, 2)
    assert p.percent == 0.0
    p.complete_candidate()
    # one completed model across two folds, of eight models * two folds
    assert p.percent == 100.0 / (8 * 2) * 2


def test_heartbeat_updates_progress_message(monkeypatch):
    import core.nps_predictor.trainer as trainer_mod

    X, y = _make_cv_data()
    dates = pd.Series(pd.date_range("2026-01-01", periods=len(X), freq="D"))
    progress = TrainingProgress()

    heartbeats = []

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout, heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        for _ in range(3):
            if heartbeat is not None:
                heartbeat()
                heartbeats.append(1)
        return _ok_result(Xtr, ytr, Xva, yva)

    monkeypatch.setattr(
        trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess
    )

    predictor = _fake_predictor()
    predictor.config.cv_timeout = 5.0
    predictor.config.cv_folds = 2

    trainer_mod.rolling_origin_train(predictor, X, y, dates=dates, progress=progress)

    assert heartbeats  # heartbeat was exercised
    snap = progress.snapshot()
    assert snap["total_models"] == 8


def test_cv_log_lines_are_emitted(caplog, monkeypatch):
    """Per-fold START/DONE/SCORE lines with elapsed time are logged."""
    import logging

    import core.nps_predictor.trainer as trainer_mod

    X, y = _make_cv_data()
    dates = pd.Series(pd.date_range("2026-01-01", periods=len(X), freq="D"))

    # One candidate so the real-subprocess log path is exercised quickly.
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: {"Ridge": Ridge()},
    )

    with caplog.at_level(logging.INFO, logger="core.nps_predictor.trainer"):
        predictor = _fake_predictor()
        trainer_mod.rolling_origin_train(
            predictor, X, y, dates=dates, progress=None
        )

    records = [r.getMessage() for r in caplog.records]
    start_lines = [r for r in records if "START name=" in r]
    score_lines = [r for r in records if "SCORE=" in r]
    assert start_lines  # every fold logs START
    assert score_lines  # every completed fold logs SCORE
    assert any("elapsed=" in r for r in records)


# ---------------------------------------------------------------------------
# Selection sample / folds are kept cheap
# ---------------------------------------------------------------------------

def test_selection_sample_and_folds_stay_cheap():
    from core.nps_predictor.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.sample_size <= 1000
    assert DEFAULT_CONFIG.cv_folds <= 2
    assert DEFAULT_CONFIG.cv_n_jobs in (1, 2)
    assert DEFAULT_CONFIG.cv_timeout >= 0


class _FakePredictor:
    def __init__(self):
        self.config = None
        self._all_models = {}
        self.model = None
        self.model_name = None
        self.algorithm_performance = {}
        self.algorithm_bucket_mae = {}
        self.cv_timing = {}
        self._training_dates = None
