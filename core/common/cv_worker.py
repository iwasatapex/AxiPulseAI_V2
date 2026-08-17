"""Generic subprocess worker for a single CV candidate fold.

The parent training process launches this module by file path so a candidate
fold (fit + predict + metric) runs in full isolation. The parent feeds a
pickled payload on stdin. The result dict is written back as a pickle on a
dedicated binary-safe temp file (``payload["result_path"]``), keeping
stdout/stderr free for library logging. If the
fold overruns its configured timeout the parent simply SIGKILLs this process
-- sklearn / CatBoost / XGBoost / LightGBM state in the parent is never
touched, so a hung candidate can never corrupt the main training run.

Supported metric modes (selected via ``payload["metric"]``):
  * "mae"  -> single-output MAE (Operation Health rolling-origin selection)
  * "nps"  -> 11-score-bucket NPS MAE + bucket MAE (NPS selection)

This module deliberately does NOT import ``core.nps_predictor`` / the whole
predictor (which would pull in catboost/xgboost/lightgbm and cost seconds of
import time per spawned fold). The NPS metrics module is loaded by absolute
path via importlib only when an NPS fold requests it.
"""
import importlib.util
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone as _sk_clone
from sklearn.metrics import mean_absolute_error


# Make ``core`` importable in the child so any model referencing a ``core``
# module (e.g. test helpers) can be unpickled. Production folds do not import
# ``core``, so per-fold startup stays cheap.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_NPS_COMPUTE = None


def _nps_compute():
    """Lazily load compute_nps_error() from the NPS metrics module."""
    global _NPS_COMPUTE
    if _NPS_COMPUTE is None:
        metrics_path = (
            Path(__file__).parent.parent / "nps_predictor" / "metrics.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_nps_metrics_standalone",
            str(metrics_path),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NPS_COMPUTE = module.compute_nps_error
    return _NPS_COMPUTE


def _worker_peak_rss_mb():
    """Return the worker's peak RSS in MiB (best effort)."""
    try:
        import resource

        # Linux reports KB for ru_maxrss.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001 - not available
        return None


def _run_fold_payload(payload):
    """Execute one candidate fold. Returns a plain dict; never raises."""
    start = time.monotonic()
    try:
        name = payload["name"]
        model = payload["model"]
        X_train = payload["X_train"]
        y_train = payload["y_train"]
        X_val = payload["X_val"]
        y_val = payload["y_val"]
        metric = payload.get("metric", "mae")

        m = _sk_clone(model)

        if name in {"XGBoost", "LightGBM", "CatBoost"}:
            # XGBoost and LightGBM are wrapped in MultiOutputRegressor. Routing
            # a full 2-D eval_set through that wrapper hands each 1-output
            # sub-estimator the entire 11-column validation target, which breaks
            # both (LightGBM "Wrong type for label"; XGBoost "Invalid
            # base_score ... n_targets"). The final full-data fit uses no
            # eval_set, so fitting these without eval_set keeps CV selection
            # consistent with the final refit. CatBoost is native multi-output
            # (MultiRMSE) and accepts a 2-D eval_set directly.
            if name == "CatBoost":
                m.fit(X_train, y_train, eval_set=(X_val, y_val))
            else:
                m.fit(X_train, y_train)
        else:
            m.fit(X_train, y_train)

        pred = m.predict(X_val)

        if metric == "nps":
            nps_mae = float(_nps_compute()(y_val, pred))
            bucket_mae = float(mean_absolute_error(y_val, pred))
            return {
                "status": "ok",
                "nps_mae": nps_mae,
                "bucket_mae": bucket_mae,
                "elapsed": time.monotonic() - start,
                "peak_rss_mb": _worker_peak_rss_mb(),
                "worker_pid": _worker_pid(),
            }

        score = float(mean_absolute_error(y_val, pred))
        return {
            "status": "ok",
            "score": score,
            "elapsed": time.monotonic() - start,
            "peak_rss_mb": _worker_peak_rss_mb(),
            "worker_pid": _worker_pid(),
        }
    except Exception as exc:  # noqa: BLE001 - worker must never raise
        return {
            "status": "error",
            "error": repr(exc),
            "elapsed": time.monotonic() - start,
            "worker_pid": _worker_pid(),
        }


def _worker_pid():
    """Return the current worker process PID."""
    try:
        import os

        return os.getpid()
    except Exception:  # noqa: BLE001
        return None


def _cv_worker_main():
    data = pickle.loads(sys.stdin.buffer.read())
    result = _run_fold_payload(data)
    result_path = data.get("result_path")
    if result_path:
        # Binary-safe result channel: a dedicated temp file owned by the
        # parent. stdout/stderr stay free for library logging (e.g. XGBoost's
        # per-iteration progress, which goes to stdout) and can never corrupt
        # the serialized result.
        with open(result_path, "wb") as fh:
            pickle.dump(result, fh)
    else:
        # Fallback for direct invocation without a result_path.
        sys.stdout.buffer.write(pickle.dumps(result))
        sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(_cv_worker_main())
