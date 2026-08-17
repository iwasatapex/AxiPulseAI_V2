"""
Focused regression tests for the CV subprocess serialization protocol.

Catches the failure where a fold estimator writes text to stdout during fit
(XGBoost emits ``[0]\\tvalidation_0-rmse:...`` progress there) which used to
contaminate the pickled result stream and fail with::

    UnpicklingError("invalid load key, '['.")

The result now travels on a dedicated binary-safe temp file channel, so
stdout/stderr are free for library logging.
"""
import numpy as np
import pandas as pd
import pytest

from core.common.cv_runner import evaluate_fold_in_subprocess
from core.nps_predictor.cv_test_helpers import SlowEstimator, StdoutNoiseEstimator

xgboost = pytest.importorskip("xgboost")


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, 5)).astype("float32"),
        columns=[f"f{i}" for i in range(5)],
    )
    y = pd.Series(rng.uniform(0.0, 100.0, n), dtype=np.float32)
    return X, y


def test_xgboost_fold_runs_through_subprocess_runner():
    X, y = _data()
    model = xgboost.XGBRegressor(
        n_estimators=10, max_depth=3, n_jobs=1, verbosity=0
    )
    res = evaluate_fold_in_subprocess(
        "XGBoost", model,
        X.iloc[:90], y[:90], X.iloc[90:], y[90:],
        timeout=60, metric="mae",
    )
    assert res["status"] == "ok"
    assert "score" in res
    assert isinstance(res["score"], float)
    assert res["score"] >= 0.0
    assert isinstance(res["elapsed"], float)


def test_stdout_noise_cannot_corrupt_result():
    """An estimator writing text to stdout must not corrupt the result."""
    X, y = _data()
    res = evaluate_fold_in_subprocess(
        "Noisy", StdoutNoiseEstimator(),
        X.iloc[:90], y[:90], X.iloc[90:], y[90:],
        timeout=60, metric="mae",
    )
    assert res["status"] == "ok"
    assert isinstance(res["score"], float)
    assert isinstance(res["elapsed"], float)


def test_timeout_still_sigkills():
    X, y = _data()
    import time

    t0 = time.monotonic()
    res = evaluate_fold_in_subprocess(
        "Slow", SlowEstimator(delay=30),
        X.iloc[:90], y[:90], X.iloc[90:], y[90:],
        timeout=0.5, metric="mae",
    )
    assert res["status"] == "timeout"
    assert time.monotonic() - t0 < 10.0
