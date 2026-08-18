"""
AxiPulseAI – NPS Training Pipeline
11-output score-distribution prediction with NPS-level model selection.

Key guarantees:
- Full training rows are preserved; repeated dates are NOT aggregated.
- NPS model selection uses actual NPS MAE, not bucket-count MAE.
- Bucket-count MAE is retained as a secondary diagnostic.
- Temporal CV splits on DISTINCT DATES, never through the middle of a date.
- Selection uses a bounded, deterministic, temporally representative sample.
- Only the selected model receives the final full-data refit.
- Sample-fitted candidate models are released BEFORE the full-data refit;
  their CV scores remain as leaderboard metadata in algorithm_performance /
  algorithm_bucket_mae. They are not required for normal inference.
- Final full-data refit is serial and occurs exactly once.
- A failed final refit fails the training operation; it never reports trained=True.
- Large temporary frames are explicitly released before final refit.
- RSS instrumentation is available only when config.verbose=True.
"""

import gc
import logging
import contextlib
import time
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from .feature_engineering import prepare_features
from .preprocessing import (
    compute_feature_stats,
    impute_missing,
    clip_outliers_iqr,
)
from .models import create_model_registry
from .resource_guard import apply_final_cpu_config, guard_final_fit, final_fit_feasible
from .constants import MODEL_VERSION
from .metrics import compute_nps_error
from ..common.temporal_dataset import shift_target_next_day, tail_by_distinct_dates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CV subprocess evaluation (shared with Operation Health)
#
# A single candidate fold (fit + predict + metric) runs inside an isolated
# subprocess via core.common.cv_runner / core.common.cv_worker. This is the
# ONLY safe way to enforce a hard CV timeout without leaving sklearn / CatBoost
# / XGBoost / LightGBM state corrupted in the main training process: the child
# can be SIGKILLed on timeout and the parent simply discards that candidate. No
# signal/alarm is ever raised in-process.
# ---------------------------------------------------------------------------

def _evaluate_fold_in_subprocess(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    timeout,
    heartbeat=None,
    memory_ceiling_mb=None,
    on_spawn=None,
):
    """Run a single candidate fold in a subprocess with a hard timeout.

    NPS mode: returns one of
      {"status": "ok",           "nps_mae", "bucket_mae", "elapsed",
                                 "peak_rss_mb", "worker_pid"}
      {"status": "error",        "error", "elapsed"}
      {"status": "timeout",      "timeout", "elapsed"}
      {"status": "memory_limit", "ceiling_mb", "peak_rss_mb", "elapsed"}

    ``memory_ceiling_mb`` arms the hard per-fold RAM guard. ``on_spawn`` is
    invoked with the worker PID right after the subprocess is spawned.
    """
    from ..common.cv_runner import evaluate_fold_in_subprocess as _run

    return _run(
        name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        timeout,
        metric="nps",
        heartbeat=heartbeat,
        memory_ceiling_mb=memory_ceiling_mb,
        on_spawn=on_spawn,
    )


# ---------------------------------------------------------------------------
# Memory instrumentation
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Return current process RSS in MB when available."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass

    try:
        import resource

        # Linux reports KB for ru_maxrss.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def _log_rss(predictor, label: str) -> None:
    """Log RSS only when verbose mode is enabled."""
    if getattr(predictor.config, "verbose", False):
        logger.info(
            "[rss] %-48s %8.0f MB",
            label,
            _rss_mb(),
        )


def _available_ram_mb() -> float:
    """Return available system RAM in MiB (best effort), else NaN."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


def _release_cv_resources() -> None:
    """Release worker/process resources accumulated during CV before refit.

    The subprocess fold workers are already reaped per fold inside
    cv_runner. This forces a collection of any pickled-payload buffers and
    candidate references still resident in the parent before the peak-RAM
    full-data refit begins.
    """
    for _ in range(3):
        gc.collect()


# ---------------------------------------------------------------------------
# Joblib / tqdm bridge
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib callbacks so tqdm advances as jobs complete."""
    class TqdmBatchCompletionCallback(
        joblib.parallel.BatchCompletionCallBack
    ):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def _emit(progress, stage, message=None, percent=None):
    """Best-effort progress emission; never raises during training."""
    if progress is None:
        return
    try:
        progress.set_stage(stage, message=message, percent=percent)
    except Exception:  # pragma: no cover - progress is advisory only
        pass


def _emit_cv_log(fmt, *args):
    """Emit a CV stage log line at INFO with the standard prefix."""
    logger.info("CV %s", fmt % args)


def train_nps_predictor(
    predictor,
    filepath: str,
    tune: bool = False,
    progress=None,
):
    """
    Train the NPS predictor.

    Lifecycle:

        load full data
        -> engineer features
        -> build temporal targets
        -> build X/y
        -> bounded temporal model selection
        -> release selection objects AND sample-fitted candidates
        -> FULL REFIT ONLY THE WINNER, ONCE, SERIAL
        -> calculate final metadata
        -> release X/y
        -> garbage collect

    The full dataset is never sampled for the final refit.
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------

    _emit(progress, "loading", message=f"Loading dataset {filepath}")
    df = load_data(predictor, filepath)

    # History buffer is DAY-based (the last N distinct calendar dates),
    # not "N arbitrary rows" — essential for repeated-date simulation data.
    predictor._history_buffer = tail_by_distinct_dates(
        df,
        date_col="date",
        days=predictor.config.history_buffer_days,
    )

    _log_rss(predictor, "1. load_data complete")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------

    # df is exclusively owned by this function and is not reused after
    # feature engineering.
    _emit(progress, "feature_engineering", message="Engineering features")
    features = prepare_features(
        df,
        predictor.config,
        copy=False,
    )

    # Resolve trajectory identity from the raw frame BEFORE releasing it.
    # ``features`` preserves df's row count/order, so these ids align with the
    # feature rows passed to the temporal helper below.
    trajectory_ids = _resolve_trajectory_ids(df)

    del df
    gc.collect()

    _log_rss(
        predictor,
        "2. features prepared, raw df released",
    )

    # ------------------------------------------------------------------
    # 3. Validate score-bucket targets
    # ------------------------------------------------------------------

    score_cols = [
        f"score_{i}"
        for i in range(predictor.config.num_score_buckets)
    ]

    if not all(c in features.columns for c in score_cols):
        raise ValueError(
            "Training data must contain score_0..score_10 "
            "columns for distribution prediction."
        )

    # ------------------------------------------------------------------
    # 4. Temporal target alignment
    # ------------------------------------------------------------------

    # The temporal helper is responsible for ensuring:
    #
    #     prediction_cutoff < target_time
    #
    # and for handling repeated dates according to the current project's
    # repeated-date alignment contract.
    _emit(progress, "preparing_targets", message="Aligning temporal targets")
    shifted_scores, target_times = shift_target_next_day(
        features[score_cols],
        features["date"],
        trajectory_ids=trajectory_ids,
        field_name="NPS score distribution",
    )

    has_target = shifted_scores.notna().all(axis=1)

    # Retain only rows with a valid future target.
    features = features.loc[has_target].copy()

    y = shifted_scores.loc[has_target].values.astype(np.float32)

    # Keep dates aligned with the surviving feature rows.
    dates_for_sampling = (
        features["date"]
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 5. Training metadata
    # ------------------------------------------------------------------

    predictor.training_rows = int(len(features))

    predictor.history_days = int(
        features["date"].nunique()
    )

    predictor._target_means = np.mean(
        y,
        axis=0,
    )

    if y.ndim != 2:
        raise ValueError(
            f"NPS target must be 2D; got shape={y.shape}"
        )

    if y.shape[1] != predictor.config.num_score_buckets:
        raise ValueError(
            "Unexpected number of NPS score buckets: "
            f"{y.shape[1]} != {predictor.config.num_score_buckets}"
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "NPS target contains non-finite values."
        )

    if np.any(y < 0):
        raise ValueError(
            "NPS score-bucket targets cannot be negative."
        )

    # ------------------------------------------------------------------
    # 6. Build feature matrix
    # ------------------------------------------------------------------

    _emit(progress, "preparing_features", message="Building feature matrix")
    excluded_columns = [
        "date",
        *score_cols,
        "promoter_pct",
        "passive_pct",
        "detractor_pct",
        "nps_today",
    ]

    X = features.drop(
        columns=excluded_columns,
        errors="ignore",
    )

    # X is newly owned here. Sanitize in place to avoid unnecessary
    # full-size copies for a 1M-row dataset.
    X.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    predictor._feature_stats = compute_feature_stats(X)

    X = impute_missing(
        X,
        predictor._feature_stats,
        copy=False,
    )

    X = X.astype(np.float32)

    if predictor.config.clip_outliers:
        X = clip_outliers_iqr(
            X,
            predictor._feature_stats,
            copy=False,
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values "
            "after preprocessing."
        )

    predictor.feature_names = list(X.columns)

    logger.info(
        "Prepared %d features (%d training rows, %d distinct dates).",
        len(predictor.feature_names),
        predictor.training_rows,
        predictor.history_days,
    )

    # ------------------------------------------------------------------
    # 7. Validate total survey counts when present
    # ------------------------------------------------------------------

    score_sum = features[score_cols].sum(axis=1)

    if "total_surveys" in features.columns:
        if not np.allclose(
            score_sum.to_numpy(),
            features["total_surveys"].to_numpy(),
            equal_nan=True,
        ):
            raise ValueError(
                "score_0..score_10 totals do not match total_surveys."
            )

    # ------------------------------------------------------------------
    # 8. Release large feature-engineering intermediates
    # ------------------------------------------------------------------

    del features
    del shifted_scores
    del has_target
    del target_times
    del score_sum
    del trajectory_ids

    gc.collect()

    _log_rss(
        predictor,
        "3. X/y built, intermediates released",
    )

    # ------------------------------------------------------------------
    # 9. Bounded temporal selection sample
    # ------------------------------------------------------------------

    if (
        predictor.config.sample_for_selection
        and predictor.training_rows > predictor.config.sample_size
    ):
        (
            X_sample,
            y_sample,
            dates_sample,
        ) = _select_temporal_sample(
            X,
            y,
            dates_for_sampling,
            predictor.config.sample_size,
            predictor.config.random_state,
        )

        logger.info(
            "Selected %d temporally representative rows for CV/selection "
            "across %d distinct dates.",
            len(X_sample),
            dates_sample.nunique(),
        )
    else:
        X_sample = X
        y_sample = y
        dates_sample = dates_for_sampling

    # dates_for_sampling is no longer needed after creating the sample.
    del dates_for_sampling
    gc.collect()

    # ------------------------------------------------------------------
    # 10. Model selection
    # ------------------------------------------------------------------

    _emit(progress, "model_selection", message="Selecting best model (CV)")

    cold_start_threshold = getattr(
        predictor.config,
        "cold_start_threshold",
        30,
    )

    if predictor.history_days < cold_start_threshold:
        logger.info(
            "Cold-start mode: %d distinct dates < threshold %d.",
            predictor.history_days,
            cold_start_threshold,
        )

        cold_start_train(
            predictor,
            X_sample,
            y_sample,
            progress=progress,
        )

    else:
        logger.info(
            "Rolling-origin model-selection mode: %d distinct dates.",
            predictor.history_days,
        )

        rolling_origin_train(
            predictor,
            X_sample,
            y_sample,
            dates=dates_sample,
            tune=tune,
            progress=progress,
            full_rows=int(len(X)),
            full_cols=int(X.shape[1]),
        )

    _log_rss(
        predictor,
        "4. candidate CV + best-model selection complete",
    )

    # ------------------------------------------------------------------
    # 11. Release bounded selection objects
    # ------------------------------------------------------------------

    if X_sample is not X:
        del X_sample

    if y_sample is not y:
        del y_sample

    del dates_sample

    gc.collect()

    _log_rss(
        predictor,
        "5. selection sample released",
    )

    # ------------------------------------------------------------------
    # 11b. Release sample-fitted candidate models BEFORE the full-data refit
    # ------------------------------------------------------------------

    # The 1M-row final refit is the peak-RAM phase of training. The
    # candidate models fitted on the bounded selection sample must NOT stay
    # resident during that refit.
    #
    # These candidates are not required for normal inference:
    #   - predict_single() uses predictor.model directly unless the opt-in
    #     ensemble_weights are populated;
    #   - predict_ensemble() / predict_leaderboard() are opt-in surfaces.
    # Their CV performance is retained as leaderboard metadata in
    # predictor.algorithm_performance / predictor.algorithm_bucket_mae.
    if predictor._all_models:
        logger.info(
            "Releasing %d sample-fitted candidate model(s) before "
            "full-data refit.",
            len(predictor._all_models),
        )

        predictor._all_models = {}

    gc.collect()

    _log_rss(
        predictor,
        "5c. candidate models released before full-data refit",
    )

    # ------------------------------------------------------------------
    # 11c. Release all CV worker/process resources before the full-data refit.
    #
    # Every fold subprocess is already reaped inside cv_runner. This drains any
    # leftover pickled-payload / candidate references so the peak-RAM refit
    # starts from a minimal footprint. It does NOT drop the winner clone.
    # ------------------------------------------------------------------
    _release_cv_resources()

    # ------------------------------------------------------------------
    # 12. Final full-data refit
    # ------------------------------------------------------------------

    # The normal rolling-origin path leaves predictor.model as an
    # UNFIT clone of the selected algorithm.
    #
    # The final full-data fit happens exactly once here.
    #
    # No Parallel().
    # No fitting every candidate.
    # No second fit of the winner.
    if predictor.model is None:
        raise RuntimeError(
            "Model selection did not produce a selected model suitable "
            "for final full-data refit."
        )

    selected_name = predictor.model_name

    logger.info(
        "Full-data refit of selected NPS model '%s' on %d rows.",
        selected_name,
        len(X),
    )

    # Log the final-refit memory context BEFORE fitting: full rows, X dtype,
    # X shape, and available system RAM. This is the peak-RAM phase of
    # training; the line is emitted even when verbose is off.
    logger.info(
        "Final refit context: full_rows=%d X_dtype=%s X_shape=%s "
        "available_ram=%.0fMB",
        len(X),
        getattr(X, "dtypes", None) if hasattr(X, "dtypes") else (
            X.dtype if hasattr(X, "dtype") else "n/a"
        ),
        list(X.shape),
        _available_ram_mb(),
    )

    # ------------------------------------------------------------------
    # Optional GPU acceleration for the FINAL full-data fit only.
    #
    # CV / model selection above always run on CPU (no GPU memory is
    # allocated there). GPU is opt-in via config.use_gpu and is applied
    # ONLY to the selected model right before this single final fit.
    # If the GPU is unavailable or the model family does not support it,
    # the fit falls back to CPU automatically and never crashes.
    # ------------------------------------------------------------------
    gpu_requested = bool(getattr(predictor.config, "use_gpu", False))
    gpu_enabled = False
    fit_device = "cpu"

    if gpu_requested:
        from .gpu import (
            apply_gpu_params,
            gpu_available,
            gpu_memory_info,
            select_final_fit_device,
        )

        logger.info("GPU requested (config.use_gpu=True).")

        if gpu_available():
            logger.info("GPU available: %s.", gpu_memory_info() or "detected")
        else:
            logger.info(
                "GPU requested but not available; falling back to CPU."
            )

        device = select_final_fit_device(selected_name, predictor.config)
        logger.info(
            "Selected algorithm: %s. Execution device: %s.",
            selected_name,
            device,
        )

        if device == "gpu":
            try:
                gpu_enabled = apply_gpu_params(
                    predictor.model,
                    selected_name,
                    predictor.config,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "GPU setup failed (%s); falling back to CPU.",
                    exc,
                )
                gpu_enabled = False

            if gpu_enabled:
                fit_device = "gpu"
                logger.info(
                    "Final fit running on GPU for %s (%s).",
                    selected_name,
                    gpu_memory_info() or "VRAM n/a",
                )
            else:
                logger.info(
                    "Final fit staying on CPU for %s.",
                    selected_name,
                )
    else:
        logger.debug("GPU disabled (config.use_gpu=False); final fit on CPU.")

    # Report the final-fit stage BEFORE the fit so the GUI never looks frozen.
    _emit(
        progress,
        "final_refit",
        message=(
            f"Final training: {selected_name} — {fit_device.upper()} "
            f"on {len(X):,} rows..."
        ),
    )
    if progress is not None:
        try:
            progress.set_final_fit(
                selected_name,
                device=fit_device,
                rows=int(len(X)),
            )
        except Exception:  # pragma: no cover - advisory
            pass

    _log_rss(
        predictor,
        "5b. immediately before final full-data refit",
    )

    # ------------------------------------------------------------------
    # Final-fit resource guard.
    #
    # Runs ONLY on the selected model, immediately before the single
    # full-data fit. It forces safe serial CPU parallelism (n_jobs=1,
    # outer + inner) and, for tree ensembles, estimates the footprint and
    # either downscales (opt-in) or raises a clear diagnostic when it cannot
    # fit under final_fit_memory_budget_mb. It NEVER drops rows, aggregates,
    # or substitutes a different model. If the guard raises, training fails
    # cleanly instead of letting the OS OOM-kill the machine.
    # ------------------------------------------------------------------
    try:
        guard_final_fit(predictor, X, y, device=fit_device)
    except RuntimeError:
        predictor.trained = False
        predictor.model = None
        gc.collect()
        raise

    try:
        predictor.model.fit(
            X,
            y,
        )
    except Exception as exc:
        # A failed GPU fit must not lose the training operation. Retry the
        # final fit once on CPU (if GPU was requested and active). The
        # normal path still performs exactly one final fit.
        if gpu_enabled:
            logger.warning(
                "GPU final fit failed for %s (%s); retrying on CPU.",
                selected_name,
                exc,
            )
            try:
                from .gpu import _GPU_ELIGIBLE

                if selected_name in _GPU_ELIGIBLE:
                    if selected_name == "CatBoost":
                        est = getattr(predictor.model, "estimator", predictor.model)
                        est.set_params(task_type="CPU", devices=None)
                    elif selected_name == "XGBoost":
                        predictor.model.estimator.set_params(
                            device="cpu", tree_method="hist", n_jobs=1
                        )
                    elif selected_name == "LightGBM":
                        predictor.model.estimator.set_params(
                            device="cpu", n_jobs=1
                        )
                # The GPU path bypassed the RAM guard (device!="cpu"
                # short-circuits). Falling back to CPU now re-verifies the
                # full-data CPU fit is feasible under final_fit_memory_budget_mb
                # BEFORE fitting; otherwise the 1M-row CPU fit could OOM.
                guard_final_fit(predictor, X, y, device="cpu")
                predictor.model.fit(X, y)
                gpu_enabled = False
                logger.warning(
                    "Final full-data refit succeeded on CPU fallback for %s.",
                    selected_name,
                )
            except Exception as fallback_exc:
                predictor.trained = False
                predictor.model = None
                gc.collect()
                logger.exception(
                    "Final full-data refit failed for model '%s' "
                    "(CPU fallback also failed).",
                    selected_name,
                )
                raise RuntimeError(
                    f"Final full-data refit failed for selected NPS model "
                    f"'{selected_name}'."
                ) from fallback_exc
        else:
            # NEVER mark a failed final refit as trained.
            predictor.trained = False
            predictor.model = None

            gc.collect()

            logger.exception(
                "Final full-data refit failed for model '%s'.",
                selected_name,
            )

            raise RuntimeError(
                f"Final full-data refit failed for selected "
                f"NPS model '{selected_name}'."
            ) from exc

    # Minimal model retention: only the full-data-fitted winner is kept.
    # Candidate models are NOT recreated after the refit; the leaderboard
    # retains their CV performance as metadata only.
    predictor._all_models = {selected_name: predictor.model}
    predictor.trained = True

    logger.info(
        "Final full-data refit complete: %s",
        selected_name,
    )

    if progress is not None:
        try:
            progress.set_final_fit(
                selected_name,
                device=fit_device,
                rows=int(len(X)),
            )
        except Exception:  # pragma: no cover - advisory
            pass

    _log_rss(
        predictor,
        "6. full-data refit complete",
    )

    # ------------------------------------------------------------------
    # 13. Training metadata
    # ------------------------------------------------------------------

    # Record the runtime + dependency versions the model was trained under, so
    # a production artifact carries truthful, verifiable runtime provenance
    # (e.g. Python 3.13.x). Never fabricated.
    import importlib.metadata as _imeta
    import platform as _platform
    lib_versions = {
        "python": _platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": _imeta.version("scikit-learn"),
        "joblib": joblib.__version__,
    }
    for _lib in ["xgboost", "lightgbm", "catboost", "shap", "pyarrow"]:
        try:
            lib_versions[_lib] = _imeta.version(_lib)
        except Exception:
            lib_versions[_lib] = None

    predictor.metadata = {
        "engine_version": MODEL_VERSION,
        "model_name": predictor.model_name,
        "algorithm": predictor.model_name,
        "training_rows": predictor.training_rows,
        "history_days": predictor.history_days,
        "feature_count": len(predictor.feature_names),
        "predict_mode": "distribution",
        "num_scores": predictor.config.num_score_buckets,
        "library_versions": lib_versions,
    }

    # Persist resource-aware selection diagnostics (final-fit feasibility) as
    # per-candidate metadata. Never fabricated: only computed values from the
    # selection stage are recorded.
    if getattr(predictor, "model_selection_diagnostics", None):
        predictor.metadata["model_selection_diagnostics"] = predictor.model_selection_diagnostics

    predictor.training_stats = {
        "rows": int(len(X)),
        "training_rows": int(predictor.training_rows),
        "history_days": int(predictor.history_days),
        "features": int(X.shape[1]),
        "outputs": int(y.shape[1]),
        "target_mean": y.mean(axis=0).tolist(),
        "target_std": y.std(axis=0).tolist(),
    }

    # ------------------------------------------------------------------
    # 14. SHAP / feature importance
    # ------------------------------------------------------------------

    # Full-scale SHAP can be extremely expensive at 1M rows.
    if len(X) > 50000:
        predictor.config.enable_shap = False

    predictor._compute_feature_importance()

    # Intentionally disabled for the full-scale training matrix.
    # If SHAP is re-enabled later, it must operate on a small bounded sample.
    #
    # if predictor.config.enable_shap:
    #     X_sample_shap = X.sample(
    #         min(100, len(X)),
    #         random_state=predictor.config.random_state,
    #     )
    #     compute_shap(predictor, X_sample_shap)
    #     del X_sample_shap

    # ------------------------------------------------------------------
    # 15. Release full training matrix
    # ------------------------------------------------------------------

    del X
    del y

    gc.collect()

    _log_rss(
        predictor,
        "7. X/y released",
    )

    logger.info(
        "Training complete. Model=%s rows=%d distinct_dates=%d.",
        predictor.model_name,
        predictor.training_rows,
        predictor.history_days,
    )


# ---------------------------------------------------------------------------
# Deterministic temporally representative selection sample
# ---------------------------------------------------------------------------

def _select_temporal_sample(
    X,
    y,
    dates,
    sample_size,
    seed,
):
    """
    Return a deterministic sample spanning the full temporal range.

    The sample:
    - contains <= sample_size rows
    - spans the available date range
    - preserves chronological order
    - does not shuffle rows
    - does not draw future information
    - is deterministic for the same input
    - is used only for model selection/CV
    - NEVER replaces the full-data final refit

    ``seed`` is retained for API compatibility and reproducibility
    documentation. This implementation is deterministic and does not
    require random sampling.
    """

    del seed  # intentionally deterministic

    dates = pd.Series(
        pd.to_datetime(dates)
    ).reset_index(drop=True)

    n = len(X)

    if n != len(y) or n != len(dates):
        raise ValueError(
            "X, y and dates must have equal lengths."
        )

    if n <= sample_size:
        return (
            X,
            y,
            dates,
        )

    # Unique dates in chronological order.
    unique_dates = (
        pd.Index(dates.dropna().unique())
        .sort_values()
    )

    n_dates = len(unique_dates)

    if n_dates < 2:
        selected_idx = np.linspace(
            0,
            n - 1,
            min(sample_size, n),
            dtype=np.int64,
        )
        selected_idx = np.unique(selected_idx)

        return (
            X.iloc[selected_idx],
            y[selected_idx],
            dates.iloc[selected_idx].reset_index(drop=True),
        )

    # We want temporal coverage, not "last N rows".
    #
    # Use at most one temporal window per requested sample row, but normally
    # far fewer windows when there are many dates. A practical cap keeps the
    # selection computation cheap.
    n_windows = min(
        100,
        sample_size,
        n_dates,
    )

    date_rank = pd.Series(
        pd.factorize(
            dates,
            sort=True,
        )[0],
        dtype=np.int64,
    ).to_numpy()

    window_of_row = (
        date_rank * n_windows
    ) // n_dates

    window_of_row = np.minimum(
        window_of_row,
        n_windows - 1,
    )

    rows_per_window = max(
        1,
        sample_size // n_windows,
    )

    selected_chunks = []

    for window_id in range(n_windows):
        idx = np.flatnonzero(
            window_of_row == window_id
        )

        if idx.size == 0:
            continue

        if idx.size <= rows_per_window:
            selected = idx
        else:
            positions = np.linspace(
                0,
                idx.size - 1,
                rows_per_window,
            )

            selected = np.unique(
                np.round(
                    positions
                ).astype(np.int64)
            )

            selected = idx[selected]

        selected_chunks.append(selected)

    if not selected_chunks:
        raise RuntimeError(
            "Temporal selection produced no rows."
        )

    selected_idx = np.concatenate(
        selected_chunks
    )

    # Restore original chronological row order.
    selected_idx = np.sort(
        np.unique(selected_idx)
    )

    # Enforce the hard size limit.
    if selected_idx.size > sample_size:
        selected_idx = selected_idx[:sample_size]

    return (
        X.iloc[selected_idx],
        y[selected_idx],
        dates.iloc[selected_idx].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_data(predictor, filepath):
    """
    Load the complete dataset.

    Repeated dates are intentional.
    One row remains one simulation sample.
    No date aggregation is performed.
    """

    from .constants import REQUIRED_COLUMNS

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    loaders = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }

    suffix = path.suffix.lower()

    if suffix not in loaders:
        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    df = loaders[suffix](path)

    required = REQUIRED_COLUMNS.union(
        {f"score_{i}" for i in range(11)}
    )

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if "date" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'date' column."
        )

    raw_dates = df["date"].copy()

    try:
        parsed = pd.to_datetime(
            raw_dates,
            errors="coerce",
            format="mixed",
        )
    except TypeError:
        parsed = raw_dates.apply(
            lambda value: pd.to_datetime(
                value,
                errors="coerce",
            )
        )

    still_bad = (
        parsed.isna()
        & raw_dates.notna()
    )

    if still_bad.any():
        try:
            retry = pd.to_datetime(
                raw_dates[still_bad],
                errors="coerce",
                dayfirst=True,
                format="mixed",
            )
        except TypeError:
            retry = raw_dates[still_bad].apply(
                lambda value: pd.to_datetime(
                    value,
                    errors="coerce",
                    dayfirst=True,
                )
            )

        parsed.loc[still_bad] = retry

    df["date"] = parsed

    if df["date"].isna().any():
        bad_mask = df["date"].isna()

        bad_rows = [
            f"row {i} -> {raw_dates.loc[i]!r}"
            for i in df.index[bad_mask][:20]
        ]

        n_bad = int(
            bad_mask.sum()
        )

        more = (
            f" (+{n_bad - len(bad_rows)} more)"
            if n_bad > len(bad_rows)
            else ""
        )

        raise ValueError(
            f"{n_bad} date value(s) could not be parsed: "
            + "; ".join(bad_rows)
            + more
        )

    # IMPORTANT:
    #
    # Do NOT:
    #
    #     df.groupby("date").agg(...)
    #
    # Every row is an independent simulation sample. Preserving repeated
    # dates is required by the 1M-row training contract.
    #
    # history_days is calculated separately from training_rows after target
    # construction.
    return df


def _resolve_trajectory_ids(df: pd.DataFrame):
    """Return a trajectory identity Series aligned to df rows, or None.

    A column is used as trajectory identity only when it is actually a
    trajectory: the same value must occur on multiple distinct dates.  Columns
    such as ``scenario_id`` are sometimes just per-row simulation labels (each
    row a distinct scenario on a single date) — using those as trajectory keys
    would produce zero valid ``T -> T+1`` pairs.  When no genuine multi-day
    trajectory exists, ``None`` is returned so the temporal helper falls back
    to its stable occurrence-based alignment.
    """
    for col in ("trajectory_id", "simulation_id", "run_id", "agent_id"):
        if col in df.columns:
            # These are explicit trajectory ids by contract.
            return df[col].reset_index(drop=True)

    if "scenario_id" in df.columns:
        spans = df.groupby(df["scenario_id"], sort=False)["date"].transform("nunique")
        if (spans > 1).any():
            return df["scenario_id"].reset_index(drop=True)

    return None



# ---------------------------------------------------------------------------
# Cold-start path
# ---------------------------------------------------------------------------

def cold_start_train(
    predictor,
    X,
    y,
    progress=None,
):
    """
    Cold-start model training.

    This path intentionally trains the candidate set on the bounded
    selection data. It does not claim a full-data winner unless the caller
    has a valid full-data refit path.

    For the normal 20-year dataset, rolling_origin_train() is expected
    because history_days is far above the cold-start threshold.
    """

    models = create_model_registry(
        predictor.config,
        cold_start=True,
        num_outputs=predictor.config.num_score_buckets,
    )

    if progress is not None:
        try:
            progress.set_models(len(models))
            progress.set_stage("model_selection", message="Cold-start training")
        except Exception:  # pragma: no cover - advisory
            pass

    for name, model in tqdm(
        models.items(),
        desc="Training models (cold-start)",
    ):
        if progress is not None:
            try:
                progress.start_candidate(name)
            except Exception:  # pragma: no cover - advisory
                pass
        try:
            model.fit(X, y)
            predictor._all_models[name] = model
        except Exception as exc:
            logger.warning(
                "Cold-start model %s failed: %s",
                name,
                exc,
            )

    if not predictor._all_models:
        raise RuntimeError(
            "No models trained in cold-start mode."
        )

    # Cold-start has no trustworthy temporal winner. Select the first
    # successful candidate as the production model and retain all candidates
    # for optional ensemble inference. This keeps the lifecycle invariant that
    # a successful training operation always leaves a concrete model.
    predictor.model_name = next(iter(predictor._all_models))
    predictor.model = predictor._all_models[predictor.model_name]
    predictor.algorithm_performance = {}
    predictor.algorithm_bucket_mae = {}
    predictor.trained = False  # caller performs the sole full-data refit

    logger.info(
        "Cold-start completed with %d candidate models.",
        len(predictor._all_models),
    )


# ---------------------------------------------------------------------------
# Date-aware temporal model selection
# ---------------------------------------------------------------------------

def _build_date_aware_splits(
    dates,
    n_splits,
):
    """
    Build TimeSeriesSplit-like folds using DISTINCT DATES.

    A given date can only occur in either:
      - the training side, or
      - the validation side

    never both.

    This is essential for repeated-date datasets. Ordinary
    TimeSeriesSplit on row indices can split thousands of rows from the same
    date between train and validation, causing same-day leakage.
    """

    dates = pd.Series(
        pd.to_datetime(dates)
    ).reset_index(drop=True)

    if dates.isna().any():
        raise ValueError(
            "Date-aware CV received missing dates."
        )

    unique_dates = (
        pd.Index(
            dates.unique()
        )
        .sort_values()
    )

    if len(unique_dates) <= 1:
        return []

    max_splits = len(unique_dates) - 1

    requested = max(
        1,
        int(n_splits),
    )

    actual_splits = min(
        requested,
        max_splits,
    )

    if actual_splits < 2:
        # One chronological holdout is still meaningful.
        train_date_count = max(
            1,
            len(unique_dates) - 1,
        )

        train_dates = unique_dates[
            :train_date_count
        ]
        val_dates = unique_dates[
            train_date_count:
        ]

        train_mask = dates.isin(
            train_dates
        ).to_numpy()

        val_mask = dates.isin(
            val_dates
        ).to_numpy()

        yield (
            np.flatnonzero(train_mask),
            np.flatnonzero(val_mask),
        )

        return

    # Run TimeSeriesSplit over DATE indices rather than row indices.
    date_positions = np.arange(
        len(unique_dates)
    )

    date_splitter = TimeSeriesSplit(
        n_splits=actual_splits
    )

    for train_date_idx, val_date_idx in date_splitter.split(
        date_positions
    ):
        train_dates = unique_dates[
            train_date_idx
        ]
        val_dates = unique_dates[
            val_date_idx
        ]

        train_mask = dates.isin(
            train_dates
        ).to_numpy()

        val_mask = dates.isin(
            val_dates
        ).to_numpy()

        train_rows = np.flatnonzero(
            train_mask
        )
        val_rows = np.flatnonzero(
            val_mask
        )

        if len(train_rows) == 0 or len(val_rows) == 0:
            continue

        # Explicit invariant: validation date must be strictly later than
        # every training date.
        if train_dates.max() >= val_dates.min():
            raise RuntimeError(
                "Temporal CV construction violated strict date ordering."
            )

        yield (
            train_rows,
            val_rows,
        )


# ---------------------------------------------------------------------------
# Rolling-origin model selection
# ---------------------------------------------------------------------------

def rolling_origin_train(
    predictor,
    X,
    y,
    dates=None,
    tune=False,
    progress=None,
    full_rows=None,
    full_cols=None,
):
    """
    Evaluate candidates using date-aware rolling-origin CV.

    Important:
    - The input X/y here should normally be the bounded selection sample.
    - No candidate is full-refit here.
    - Candidate models are fitted only on the bounded selection sample.
    - The winning model is returned as an UNFIT clone.
    - train_nps_predictor() performs the sole full-data refit.

    Resource-aware selection:
    - After each candidate's CV score is available, the candidate is checked for
      FINAL-FIT feasibility under ``final_fit_memory_budget_mb`` at the FULL
      training row count (``full_rows``/``full_cols``). Candidates whose
      estimated final-fit RAM exceeds the budget are excluded from winner
      selection with an explicit reason. The best NPS-MAE candidate among the
      feasible set wins. ``full_rows`` defaults to the bounded sample length when
      not supplied (best-effort).
    """
    del tune  # retained for API compatibility

    n = len(X)

    if n != len(y):
        raise ValueError(
            "X and y must have equal lengths."
        )

    if full_rows is None:
        full_rows = int(n)
    if full_cols is None:
        full_cols = int(X.shape[1])

    # Very small datasets use the explicit simple holdout.
    if n < 10:
        split_idx = int(
            0.8 * n
        )

        if split_idx <= 0 or split_idx >= n:
            raise ValueError(
                "Not enough rows for a valid train/validation split."
            )

        X_train = X.iloc[:split_idx]
        y_train = y[:split_idx]

        X_val = X.iloc[split_idx:]
        y_val = y[split_idx:]

        train_and_select_simple(
            predictor,
            X_train,
            y_train,
            X_val,
            y_val,
            progress=progress,
        )

        return

    if dates is not None:
        dates = pd.Series(
            pd.to_datetime(dates)
        ).reset_index(drop=True)

        if len(dates) != n:
            raise ValueError(
                "dates must have the same length as X/y."
            )

        cv_folds = max(
            1,
            int(
                getattr(
                    predictor.config,
                    "cv_folds",
                    2,
                )
            ),
        )

        split_list = list(
            _build_date_aware_splits(
                dates,
                n_splits=cv_folds,
            )
        )
    else:
        logger.warning(
            "No dates supplied to rolling_origin_train(); "
            "falling back to row-based TimeSeriesSplit. "
            "Repeated-date leakage protection is unavailable."
        )

        cv_folds = max(
            1,
            int(
                getattr(
                    predictor.config,
                    "cv_folds",
                    2,
                )
            ),
        )

        n_splits = min(
            cv_folds,
            max(2, n // 250),
        )

        splitter = TimeSeriesSplit(
            n_splits=n_splits
        )

        split_list = list(
            splitter.split(X)
        )

    if not split_list:
        raise RuntimeError(
            "Unable to construct valid temporal CV splits."
        )

    base_models = create_model_registry(
        predictor.config,
        cold_start=False,
        num_outputs=predictor.config.num_score_buckets,
    )

    # Resource policy is authoritative for CV: candidate folds already run in
    # isolated subprocesses one at a time, so force every candidate's own
    # parallelism to serial (n_jobs=1) to prevent nested thread/process
    # multiplication inside each fold worker. This makes the declared
    # ``cv_n_jobs=1`` policy match the estimators actually being fit.
    for name in list(base_models.keys()):
        apply_final_cpu_config(base_models[name], name, predictor.config)

    model_names = list(
        base_models.keys()
    )

    if progress is not None:
        try:
            progress.set_models(len(model_names))
        except Exception:  # pragma: no cover - advisory
            pass

    # ------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------

    base_timeout = float(
        getattr(
            predictor.config,
            "cv_timeout",
            60.0,
        )
    )

    # MLP is the most RAM/time-hungry candidate and does not early-stop like
    # the gradient boosters. Give it a stricter per-fold timeout when one is
    # configured, and rely on the per-fold RAM guard to exclude it (with an
    # explicit reason) rather than letting it exhaust the machine.
    mlp_timeout = getattr(
        predictor.config,
        "cv_mlp_timeout",
        None,
    )

    def _timeout_for(name: str) -> float:
        if name == "MLP" and mlp_timeout is not None:
            return float(mlp_timeout)
        return base_timeout

    # Hard per-fold RAM ceiling (MiB) enforced against each CV subprocess.
    memory_ceiling_mb = getattr(
        predictor.config,
        "cv_memory_ceiling_mb",
        None,
    )

    total_models = len(model_names)
    total_folds = len(split_list)

    # Per-run timing stats (fold times, per-model totals, timeouts, errors).
    cv_timing = {
        "timeout": base_timeout,
        "model_fold_elapsed": {},  # {name: [fold_elapsed, ...]}
        "model_total_elapsed": {},  # {name: total seconds}
        "fold_times": [],          # flat list of every completed fold's seconds
        "timeouts": [],            # [name, ...]
        "errors": [],              # [(name, error), ...]
        "memory_limits": [],       # [(name, peak_rss_mb), ...]
        "completed": [],           # [name, ...]
    }

    def evaluate_model(
        name,
        model,
        idx,
    ):
        if progress is not None:
            try:
                progress.start_candidate(name, total_models=total_models)
                progress.set_stage(
                    "model_selection",
                    message=f"Evaluating {name} ({idx}/{total_models})",
                )
            except Exception:  # pragma: no cover - advisory
                pass

        model_start = time.monotonic()
        fold_nps_maes = []
        fold_bucket_maes = []

        model_timeout = _timeout_for(name)

        _emit_cv_log(
            "CV model %d/%d fold %d/%d START name=%s",
            idx,
            total_models,
            0,
            total_folds,
            name,
        )

        for fold_number, (
            train_idx,
            val_idx,
        ) in enumerate(
            split_list,
            start=1,
        ):
            if progress is not None:
                try:
                    progress.start_fold(
                        fold_number,
                        total_folds=len(split_list),
                    )
                except Exception:  # pragma: no cover - advisory
                    pass

            X_train = X.iloc[train_idx]
            y_train = y[train_idx]

            X_val = X.iloc[val_idx]
            y_val = y[val_idx]

            sample_rows = int(len(X_train) + len(X_val))
            feature_count = int(X.shape[1])
            worker_pid = None

            def _on_spawn(pid):
                nonlocal worker_pid
                worker_pid = int(pid)

            def _heartbeat():
                if progress is not None:
                    try:
                        progress.set_stage(
                            "model_selection",
                            message=(
                                f"Evaluating {name} ({idx}/{total_models}) "
                                f"fold {fold_number}/{total_folds} ..."
                            ),
                        )
                    except Exception:  # pragma: no cover - advisory
                        pass

            # Log BEFORE the fold runs: model, fold, sample rows, features,
            # worker PID (populated by on_spawn once the subprocess is up).
            _emit_cv_log(
                "CV fold START model=%s fold=%d/%d sample_rows=%d "
                "features=%d worker_pid=%s",
                name,
                fold_number,
                total_folds,
                sample_rows,
                feature_count,
                worker_pid,
            )

            result = _evaluate_fold_in_subprocess(
                name,
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                timeout=model_timeout,
                heartbeat=_heartbeat,
                memory_ceiling_mb=memory_ceiling_mb,
                on_spawn=_on_spawn,
            )

            elapsed = float(result.get("elapsed", 0.0))
            peak_rss = float(result.get("peak_rss_mb", float("nan")))
            rss_str = (
                f"{peak_rss:.0f}MB" if peak_rss == peak_rss else "n/a"
            )
            cv_timing["model_fold_elapsed"].setdefault(name, []).append(elapsed)
            cv_timing["fold_times"].append(elapsed)

            status = result.get("status")

            if status == "ok":
                nps_mae = float(result["nps_mae"])
                bucket_mae = float(result["bucket_mae"])

                # Log AFTER the fold: elapsed, peak/last RSS, score.
                _emit_cv_log(
                    "CV fold DONE model=%s fold=%d/%d elapsed=%.2fs "
                    "peak_rss=%s SCORE=%.4f",
                    name,
                    fold_number,
                    total_folds,
                    elapsed,
                    rss_str,
                    nps_mae,
                )

                fold_nps_maes.append(nps_mae)
                fold_bucket_maes.append(bucket_mae)

            elif status == "memory_limit":
                ceiling = float(result.get("ceiling_mb", memory_ceiling_mb))
                logger.warning(
                    "CV MEMORY LIMIT: %s fold %d worker_rss=%.0fMB "
                    "> ceiling=%.0fMB. Candidate excluded with explicit "
                    "resource-limit reason.",
                    name,
                    fold_number,
                    peak_rss,
                    ceiling,
                )
                cv_timing["memory_limits"].append((name, peak_rss))

                if progress is not None:
                    try:
                        progress.set_stage(
                            "model_selection",
                            message=(
                                f"{name} fold {fold_number}/{total_folds} "
                                f"exceeded RAM ceiling ({peak_rss:.0f}MB) - "
                                f"skipped"
                            ),
                        )
                    except Exception:  # pragma: no cover - advisory
                        pass

                cv_timing["model_total_elapsed"][name] = (
                    time.monotonic() - model_start
                )
                return (name, None, None)

            elif status == "timeout":
                logger.warning(
                    "CV TIMEOUT: %s fold %d (%.2fs > %.2fs limit). "
                    "Candidate excluded from selection.",
                    name,
                    fold_number,
                    elapsed,
                    model_timeout,
                )
                cv_timing["timeouts"].append(name)

                if progress is not None:
                    try:
                        progress.set_stage(
                            "model_selection",
                            message=(
                                f"{name} fold {fold_number}/{total_folds} "
                                f"timed out after {elapsed:.1f}s - skipped"
                            ),
                        )
                    except Exception:  # pragma: no cover - advisory
                        pass

                cv_timing["model_total_elapsed"][name] = (
                    time.monotonic() - model_start
                )
                return (name, None, None)

            else:  # status == "error"
                err = result.get("error", "unknown error")
                logger.warning(
                    "CV model %s fold %d failed: %s. Candidate excluded.",
                    name,
                    fold_number,
                    err,
                )
                cv_timing["errors"].append((name, err))
                cv_timing["model_total_elapsed"][name] = (
                    time.monotonic() - model_start
                )
                return (name, None, None)

            if progress is not None:
                try:
                    progress.set_stage(
                        "model_selection",
                        message=(
                            f"Evaluated {name} fold {fold_number}/{total_folds} "
                            f"({elapsed:.1f}s)"
                        ),
                    )
                except Exception:  # pragma: no cover - advisory
                    pass

        if not fold_nps_maes:
            return (name, None, None)

        cv_timing["model_total_elapsed"][name] = (
            time.monotonic() - model_start
        )
        cv_timing["completed"].append(name)

        if progress is not None:
            try:
                progress.complete_candidate()
            except Exception:  # pragma: no cover - advisory
                pass

        _emit_cv_log(
            "CV model %d/%d DONE name=%s total_elapsed=%.2fs nps_mae=%.4f",
            idx,
            total_models,
            name,
            time.monotonic() - model_start,
            float(np.mean(fold_nps_maes)),
        )

        return (
            name,
            float(
                np.mean(
                    fold_nps_maes
                )
            ),
            float(
                np.mean(
                    fold_bucket_maes
                )
            ),
        )

    # Candidate-level CV parallelism is controlled explicitly by cv_n_jobs.
    #
    # Each candidate's individual fold evaluation still uses the existing
    # isolated/reaped subprocess path. Thread-based joblib workers are used
    # here so shared timing/progress metadata remains in the parent process
    # and the full training matrix is not duplicated through fork/loky.
    cv_n_jobs = max(
        1,
        int(getattr(predictor.config, "cv_n_jobs", 1)),
    )

    logger.info(
        "Temporal CV: %d folds, %d candidate models, n_jobs=%d, "
        "%.1fs per-fold timeout, %.0fMB per-fold RAM ceiling.",
        len(split_list),
        len(model_names),
        cv_n_jobs,
        base_timeout,
        memory_ceiling_mb if memory_ceiling_mb is not None else float("inf"),
    )

    model_items = list(base_models.items())

    # Preserve the existing progress UI while allowing the configurable
    # candidate-level Parallel execution. evaluate_model() remains the single
    # source of truth for fold isolation, timeout handling and scoring.
    with tqdm(
        total=len(model_names),
        desc="Evaluating models (CV)",
    ) as pbar:
        results = Parallel(
            n_jobs=cv_n_jobs,
            prefer="threads",
        )(
            delayed(evaluate_model)(
                name,
                model,
                idx,
            )
            for idx, (name, model) in enumerate(
                model_items,
                start=1,
            )
        )

        pbar.update(len(results))

    # ------------------------------------------------------------
    # Select best model
    # ------------------------------------------------------------

    avg_perf = {}
    avg_bucket_mae = {}

    for (
        name,
        nps_mae,
        bucket_mae,
    ) in results:
        if nps_mae is None:
            continue

        avg_perf[name] = float(
            nps_mae
        )

        avg_bucket_mae[name] = float(
            bucket_mae
        )

    del results
    gc.collect()

    predictor.cv_timing = cv_timing

    if not avg_perf:
        raise RuntimeError(
            "No model produced valid predictions during temporal CV. "
            "All candidates failed or timed out."
        )

    if cv_timing["timeouts"]:
        logger.warning(
            "CV candidates excluded due to timeout: %s",
            ", ".join(sorted(set(cv_timing["timeouts"]))),
        )
    if cv_timing["memory_limits"]:
        logger.warning(
            "CV candidates excluded due to per-fold RAM ceiling: %s",
            ", ".join(sorted({n for n, _ in cv_timing["memory_limits"]})),
        )
    if cv_timing["errors"]:
        logger.warning(
            "CV candidates excluded due to error: %s",
            ", ".join(sorted({n for n, _ in cv_timing["errors"]})),
        )

    # ------------------------------------------------------------
    # Resource-aware final-fit feasibility (deployment feasibility).
    #
    # A candidate may win CV but be unable to safely perform the FULL final
    # refit under final_fit_memory_budget_mb at the full training row count.
    # Exclude such candidates from winner selection with an explicit reason,
    # then pick the best NPS-MAE among the FEASIBLE candidates. This keeps the
    # final-fit memory guard as a second safety layer while preventing an
    # infeasible winner from being selected in the first place. All candidates
    # remain in the registry; this is deployment/resource feasibility, not
    # candidate removal.
    # ------------------------------------------------------------
    from .resource_guard import final_fit_feasible
    from .gpu import select_final_fit_device

    budget_mb = float(
        getattr(predictor.config, "final_fit_memory_budget_mb", 4096.0)
    )
    final_n_jobs = int(
        getattr(predictor.config, "final_cpu_n_jobs", 1)
    )
    use_gpu = bool(getattr(predictor.config, "use_gpu", False))

    resource_diagnostics = {}
    infeasible_reasons = {}
    feasible_names = []

    for name in base_models.keys():
        diag = {"cv_score": avg_perf.get(name), "final_fit_estimated_memory_mb": None,
                "final_fit_feasible": True, "reason_if_not_feasible": None}
        # A candidate that never produced a CV score is already excluded; it
        # cannot win, but still report its resource status best-effort.
        if name not in avg_perf:
            diag["final_fit_feasible"] = False
            diag["reason_if_not_feasible"] = "no CV score (failed/timed out/excluded)"
            resource_diagnostics[name] = diag
            continue

        device = select_final_fit_device(name, predictor.config)
        feasible, reason, fdiag = final_fit_feasible(
            name,
            base_models[name],
            rows=full_rows,
            cols=full_cols,
            n_outputs=predictor.config.num_score_buckets,
            budget_mb=budget_mb,
            n_jobs=final_n_jobs,
            device=device,
        )
        diag.update(fdiag)
        diag["final_fit_feasible"] = feasible
        diag["reason_if_not_feasible"] = reason
        diag["device"] = device
        diag["gpu_capable"] = bool(use_gpu and device == "gpu")
        resource_diagnostics[name] = diag

        if feasible:
            feasible_names.append(name)
        else:
            infeasible_reasons[name] = reason

    predictor.model_selection_diagnostics = resource_diagnostics

    for name, reason in infeasible_reasons.items():
        logger.warning(
            "%s excluded from winner selection: %s",
            name,
            reason,
        )

    if not feasible_names:
        details = "; ".join(
            f"{n} ({resource_diagnostics[n].get('reason_if_not_feasible') or 'infeasible'})"
            for n in base_models.keys()
            if n in resource_diagnostics
        )
        raise RuntimeError(
            "No NPS candidate is final-fit feasible under "
            "final_fit_memory_budget_mb=%.0fMB at %d rows. "
            "Every candidate's estimated memory/resource reason: %s. "
            "To proceed: raise final_fit_memory_budget_mb, enable "
            "final_fit_auto_downscale=True, or train on fewer rows."
            % (budget_mb, full_rows, details)
        )

    # Pick the best NPS-MAE among the FEASIBLE candidates.
    feasible_perf = {name: avg_perf[name] for name in feasible_names}
    best_name = min(
        feasible_perf,
        key=feasible_perf.get,
    )

    best_mae = avg_perf[
        best_name
    ]

    # ------------------------------------------------------------
    # Leave predictor._all_models EMPTY.
    #
    # No candidate is refit merely to populate _all_models. No production
    # caller requires fitted candidate objects:
    #   - predict_single() uses predictor.model directly unless the opt-in
    #     ensemble_weights are populated (they never are during training);
    #   - predict_ensemble() / predict_leaderboard() are opt-in surfaces and,
    #     when used after training, iterate over _all_models which the caller
    #     must populate explicitly.
    # The leaderboard only needs the CV performance (algorithm_performance /
    # algorithm_bucket_mae), which is retained as metadata below.
    # ------------------------------------------------------------

    predictor._all_models = {}

    gc.collect()

    # ------------------------------------------------------------
    # IMPORTANT:
    # Leave predictor.model UNFIT.
    #
    # The ONLY full-data fit happens in train_nps_predictor().
    # ------------------------------------------------------------

    predictor.model = clone(
        base_models[best_name]
    )

    predictor.model_name = best_name

    predictor.algorithm_performance = (
        avg_perf
    )

    predictor.algorithm_bucket_mae = (
        avg_bucket_mae
    )

    logger.info(
        "Selected %s: NPS MAE=%.4f. "
        "Full-data refit deferred to train_nps_predictor().",
        best_name,
        best_mae,
    )


# ---------------------------------------------------------------------------
# Simple train/validation selection
# ---------------------------------------------------------------------------

def train_and_select_simple(
    predictor,
    X_train,
    y_train,
    X_val,
    y_val,
    progress=None,
):
    """
    Simple fallback model-selection path.

    Uses NPS MAE as primary metric and bucket-count MAE as diagnostic.
    """

    models = create_model_registry(
        predictor.config,
        cold_start=False,
        num_outputs=predictor.config.num_score_buckets,
    )

    best_nps_mae = float("inf")
    best_name = None

    perfs = {}
    bucket_perfs = {}

    predictor._all_models = {}

    if progress is not None:
        try:
            progress.set_models(len(models))
            progress.set_stage("model_selection", message="Simple selection")
        except Exception:  # pragma: no cover - advisory
            pass

    for name, model in tqdm(
        models.items(),
        desc="Training models",
    ):
        if progress is not None:
            try:
                progress.start_candidate(name)
            except Exception:  # pragma: no cover - advisory
                pass
        try:
            if name in {
                "XGBoost",
                "LightGBM",
                "CatBoost",
            }:
                # XGBoost and LightGBM here are wrapped in
                # MultiOutputRegressor. Routing a full 2-D eval_set through that
                # wrapper would hand each 1-output sub-estimator the entire
                # 11-column validation target (LightGBM: "Wrong type for label";
                # XGBoost: "Invalid base_score ... n_targets: 11"). The final
                # full-data fit uses no eval_set at all, so fitting these without
                # eval_set keeps selection consistent with the final refit.
                # CatBoost is native multi-output (MultiRMSE) and accepts a 2-D
                # eval_set directly.
                if name == "CatBoost":
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=(
                            X_val,
                            y_val,
                        ),
                    )
                else:
                    model.fit(
                        X_train,
                        y_train,
                    )
            else:
                model.fit(
                    X_train,
                    y_train,
                )

            pred = model.predict(
                X_val
            )

            nps_mae = float(
                compute_nps_error(
                    y_val,
                    pred,
                )
            )

            bucket_mae = float(
                mean_absolute_error(
                    y_val,
                    pred,
                )
            )

            if not np.isfinite(nps_mae):
                raise ValueError(
                    f"Non-finite NPS MAE for {name}: {nps_mae}"
                )

            perfs[name] = nps_mae
            bucket_perfs[name] = bucket_mae

            predictor._all_models[
                name
            ] = model

            if nps_mae < best_nps_mae:
                best_nps_mae = nps_mae
                best_name = name

        except Exception as exc:
            logger.warning(
                "Model %s failed in simple selection: %s",
                name,
                exc,
            )

        finally:
            del pred
            gc.collect()
            if progress is not None:
                try:
                    progress.complete_candidate()
                except Exception:  # pragma: no cover - advisory
                    pass

    if best_name is None:
        raise RuntimeError(
            "No model succeeded in simple NPS model selection."
        )

    # Make the selected model an UNFIT clone so the normal caller can
    # perform exactly one final full-data fit.
    predictor.model = clone(
        models[best_name]
    )

    predictor.model_name = best_name

    predictor.algorithm_performance = perfs
    predictor.algorithm_bucket_mae = bucket_perfs

    logger.info(
        "Simple selection chose %s with NPS MAE=%.4f; "
        "full-data refit deferred to caller.",
        best_name,
        best_nps_mae,
    )