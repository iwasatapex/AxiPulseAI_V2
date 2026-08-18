"""Importable helpers used by CV timeout/progress tests.

``SlowEstimator`` lives in a real ``core`` module (not a test file) so it can
be unpickled inside the ``cv_worker`` subprocess, which exercises the genuine
SIGKILL-on-timeout path for an automated test.
"""
import time


class SlowEstimator:
    """A picklable estimator that sleeps before fitting.

    Used to deterministically force a per-fold CV timeout: a real subprocess
    worker is spawned, the fit blocks for ``delay`` seconds, and the parent
    must SIGKILL the child when the fold exceeds its timeout.
    """

    def __init__(self, delay=2.0):
        self.delay = delay

    def fit(self, X, y, **kwargs):
        time.sleep(self.delay)
        return self

    def predict(self, X, **kwargs):
        import numpy as np

        return np.zeros((len(X), 11), dtype=np.float64)

    def get_params(self, deep=True):
        return {"delay": self.delay}

    def set_params(self, **params):
        if "delay" in params:
            self.delay = params["delay"]
        return self


class StdoutNoiseEstimator:
    """A picklable estimator that writes text to stdout during fit.

    This reproduces the XGBoost behaviour that originally broke the CV
    subprocess protocol: the estimator emits a ``[0]\\t...``-style progress
    line on stdout while fitting. The result payload travels on a dedicated
    binary-safe channel, so this noise must never corrupt it.
    """

    def __init__(self, noise=b"[0]\tvalidation_0-rmse:1.08917\n"):
        self.noise = noise

    def fit(self, X, y, **kwargs):
        import os
        import sys

        data = self.noise.encode() if isinstance(self.noise, str) else self.noise
        os.write(sys.stdout.fileno(), data)
        sys.stdout.flush()
        return self

    def predict(self, X, **kwargs):
        import numpy as np

        return np.zeros(len(X), dtype=np.float64)

    def get_params(self, deep=True):
        return {"noise": self.noise}

    def set_params(self, **params):
        if "noise" in params:
            self.noise = params["noise"]
        return self


class BoomEstimator:
    """A picklable estimator whose fit raises, to exercise the error path."""

    def fit(self, X, y, **kwargs):
        raise RuntimeError("boom")

    def predict(self, X, **kwargs):
        import numpy as np

        return np.zeros((len(X), 11), dtype=np.float64)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

