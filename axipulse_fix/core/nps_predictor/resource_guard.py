"""
AxiPulseAI – Resource safety for the FINAL NPS full-data fit.

Design contract:
- The FINAL full-data refit is the peak-RAM phase of training. CV / model
  selection already run serial (cv_n_jobs=1) in reaped subprocesses and are
  unchanged here.
- CPU tree ensembles (ExtraTrees / RandomForest and the boosters) are forced to
  n_jobs=1 for the final fit so that one-tree-batch-per-core memory duplication
  can never exhaust the machine. MultiOutputRegressor's outer n_jobs and the
  inner estimator's n_jobs are BOTH set to 1, eliminating nested parallelism.
- A conservative pre-fit memory guard estimates the selected estimator's
  footprint from the real tree count / row count / output count. If that
  estimate exceeds ``config.final_fit_memory_budget_mb`` the guard:
      1) reduces safe parallelism (final_cpu_n_jobs -> 1),
      2) optionally downscales the estimator count when
         ``config.final_fit_auto_downscale`` is True (logged explicitly),
      3) otherwise RAISES a clear resource diagnostic BEFORE .fit(), so the OS
         never OOM-kills the machine.
- The guard NEVER drops rows, NEVER aggregates, NEVER swaps the model, and
  never changes candidate-selection scores.
"""

import logging

import numpy as np
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)

# Model families whose memory footprint is dominated by trees and therefore
# predictable from the tree count. MLP is intentionally excluded: its footprint
# is not tree-bounded and we do not hard-fail on it.
_TREE_MODELS = {
    "ExtraTrees",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "HistGradientBoosting",
    "GradientBoosting",
}

# Attributes, in priority order, that expose the effective tree count of an
# estimator. CatBoost uses ``iterations``; HGB uses ``max_iter``.
_TREE_COUNT_ATTRS = ("n_estimators", "iterations", "max_iter")


def _available_ram_mb() -> float:
    """Return available system RAM in MiB (best effort), else NaN."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001 - not available on all platforms
        pass
    return float("nan")


def _rss_mb() -> float:
    """Return current process RSS in MiB (best effort), else NaN."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource

        # Linux reports KB for ru_maxrss.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001
        return float("nan")


def inner_estimator(model):
    """Unwrap a MultiOutputRegressor / pipeline to the leaf estimator."""
    est = getattr(model, "estimator", model)
    return est


def tree_count(estimator):
    """Return the effective tree count of an estimator, or None if unknown."""
    for attr in _TREE_COUNT_ATTRS:
        if hasattr(estimator, attr):
            try:
                value = int(getattr(estimator, attr))
            except (TypeError, ValueError):  # noqa: BLE001
                continue
            return value
    return None


def estimate_tree_fit_memory_mb(
    name,
    estimator,
    rows,
    n_outputs,
    n_jobs=1,
    nodes_per_row=2.0,
    bytes_per_node=70.0,
):
    """Conservative estimate (MiB) of a full tree-ensemble final fit.

    Each full-depth tree holds ~ ``nodes_per_row * rows`` nodes; each node costs
    ~ ``bytes_per_node`` bytes (feature index + threshold + child pointers).
    A MultiOutputRegressor builds one forest per output column, so the total is
    multiplied by ``n_outputs``. ``n_jobs > 1`` adds a modest transient for the
    trees still being built in parallel. Returns None for non-tree estimators.
    """
    if name not in _TREE_MODELS:
        return None
    n_trees = tree_count(estimator)
    if n_trees is None:
        return None
    per_tree_bytes = float(rows) * nodes_per_row * bytes_per_node
    parallel_factor = 1.0
    if n_jobs > 1:
        parallel_factor = 1.0 + 0.25 * (float(n_jobs) - 1.0)
    return (
        n_trees
        * int(n_outputs)
        * per_tree_bytes
        * parallel_factor
        / 1e6
    )


def estimate_x_bytes_mb(rows, cols, itemsize=4):
    """Feature-matrix resident size in MiB (float32 default)."""
    return float(rows) * float(cols) * float(itemsize) / 1e6


def estimate_final_fit_total_mb(
    name,
    estimator,
    rows,
    cols,
    n_outputs,
    n_jobs=1,
    itemsize=4,
    baseline_mb=512.0,
):
    """Estimate total final-fit RAM (MiB) for a candidate at FULL row count.

    Mirrors the final-fit guard's computation exactly (tree footprint + feature
    matrix + baseline interpreter/loader) so that model selection and the final
    guard agree on feasibility. Returns ``(estimate_mb, x_bytes_mb, total_mb)``.
    ``total_mb`` is None for non-tree estimators (not hard-fail bounded).
    """
    est = inner_estimator(estimator)
    est_mb = estimate_tree_fit_memory_mb(name, est, rows, n_outputs, n_jobs=n_jobs)
    x_mb = estimate_x_bytes_mb(rows, cols, itemsize)
    if est_mb is None:
        return (None, x_mb, None)
    total_mb = est_mb + x_mb + float(baseline_mb)
    return (est_mb, x_mb, total_mb)


def final_fit_feasible(
    name,
    estimator,
    rows,
    cols,
    n_outputs,
    budget_mb,
    n_jobs=1,
    itemsize=4,
    baseline_mb=512.0,
    device="cpu",
):
    """Return ``(feasible, reason_if_not_feasible, diagnostics)`` for a candidate.

    ``diagnostics`` carries ``cv_score`` (filled by caller), 
    ``final_fit_estimated_memory_mb``, ``final_fit_feasible`` and, when
    infeasible, ``reason_if_not_feasible``. Non-tree estimators (e.g. MLP) or a
    GPU final fit are treated as not hard-fail bounded, hence feasible unless a
    specific reason is given.
    """
    diag = {
        "final_fit_estimated_memory_mb": None,
        "final_fit_feasible": True,
        "reason_if_not_feasible": None,
    }
    if device != "cpu":
        diag["final_fit_feasible"] = True
        diag["reason_if_not_feasible"] = "GPU final fit (not RAM-hard-fail bounded)"
        return True, None, diag

    est_mb, x_mb, total_mb = estimate_final_fit_total_mb(
        name, estimator, rows, cols, n_outputs, n_jobs=n_jobs,
        itemsize=itemsize, baseline_mb=baseline_mb,
    )
    if total_mb is None:
        # Non-tree estimator: not hard-fail bounded by the CPU tree guard.
        diag["final_fit_feasible"] = True
        diag["reason_if_not_feasible"] = "non-tree estimator (not RAM-hard-fail bounded)"
        return True, None, diag

    diag["final_fit_estimated_memory_mb"] = round(total_mb, 1)
    if total_mb <= float(budget_mb):
        return True, None, diag

    diag["final_fit_feasible"] = False
    reason = (
        f"estimated final fit {total_mb:,.0f} MB > budget {float(budget_mb):,.0f} MB"
    )
    diag["reason_if_not_feasible"] = reason
    return False, reason, diag


def apply_final_cpu_config(model, name, config) -> int:
    """Force safe serial (n_jobs=1) config on the selected model in place.

    Sets BOTH the outer MultiOutputRegressor n_jobs and the inner estimator's
    n_jobs / thread_count to ``config.final_cpu_n_jobs`` (min 1), preventing
    nested parallelism for the final fit. Model hyperparameters that affect
    semantics (n_estimators, depth, learning rate, outputs) are untouched.
    Returns the effective n_jobs configured.
    """
    n_jobs = int(getattr(config, "final_cpu_n_jobs", 1))
    if n_jobs < 1:
        n_jobs = 1

    outer = model
    inner = getattr(model, "estimator", model)

    if isinstance(outer, MultiOutputRegressor):
        try:
            outer.set_params(n_jobs=n_jobs)
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.warning("Could not set outer n_jobs for %s: %s", name, exc)

    if hasattr(inner, "set_params"):
        try:
            if name in {"ExtraTrees", "RandomForest", "XGBoost", "LightGBM"}:
                inner.set_params(n_jobs=n_jobs)
            elif name == "CatBoost":
                inner.set_params(thread_count=n_jobs)
        except Exception as exc:  # noqa: BLE001 - MLP/pipeline etc.
            logger.debug("Could not set CPU n_jobs for %s: %s", name, exc)

    return n_jobs


def _downscale_estimator(estimator, target_trees):
    """Reduce the estimator count on a tree estimator in place (best effort)."""
    if hasattr(estimator, "n_estimators"):
        estimator.set_params(n_estimators=max(1, int(target_trees)))
        return True
    if hasattr(estimator, "iterations"):
        estimator.set_params(iterations=max(1, int(target_trees)))
        return True
    if hasattr(estimator, "max_iter"):
        estimator.set_params(max_iter=max(1, int(target_trees)))
        return True
    return False


def _downscale_for_budget(name, estimator, rows, n_outputs, budget_mb, n_jobs):
    """Return a tree count that fits ``budget_mb``, or None if impossible."""
    per_tree_mb = estimate_tree_fit_memory_mb(
        name, estimator, rows, n_outputs, n_jobs=n_jobs
    )
    if per_tree_mb is None:
        return None
    # Derive per-(tree,output) MiB from the full estimate.
    n_trees = tree_count(estimator)
    if not n_trees:
        return None
    per_unit = per_tree_mb / n_trees
    if per_unit <= 0:
        return None
    fit = int(budget_mb // per_unit)
    if fit < 1:
        return None
    return fit


def guard_final_fit(predictor, X, y, device="cpu"):
    """Validate and (optionally) reduce resource use before the final fit.

    Emits the pre-fit report and either reduces safe parallelism, downscales
    the estimator count (opt-in), or raises a clear diagnostic when the selected
    estimator cannot fit under ``final_fit_memory_budget_mb``. Never mutates
    the dataset. Runs only for the FINAL fit of the selected model.

    ``device`` ("cpu" or "gpu") is the final-fit execution device. A GPU final
    fit places the heavy compute on VRAM, not system RAM, so the hard-fail RAM
    guard is only applied when the final fit runs on CPU. The serial n_jobs=1
    configuration is applied in both cases.
    """
    config = predictor.config
    name = predictor.model_name
    budget_mb = float(getattr(config, "final_fit_memory_budget_mb", 4096.0))

    rows = int(len(X))
    cols = int(X.shape[1])
    try:
        itemsize = int(X.dtypes.iloc[0].itemsize)
    except Exception:  # noqa: BLE001 - numpy array
        itemsize = int(getattr(getattr(X, "dtype", np.dtype("float32")), "itemsize", 4))

    x_bytes_mb = estimate_x_bytes_mb(rows, cols, itemsize)

    # 1) Reduce safe parallelism FIRST (never touches rows/model).
    n_jobs = apply_final_cpu_config(predictor.model, name, config)

    est = inner_estimator(predictor.model)
    n_trees = tree_count(est)
    estimate_mb = estimate_tree_fit_memory_mb(
        name, est, rows, y.shape[1], n_jobs=n_jobs
    )

    available_mb = _available_ram_mb()

    logger.info(
        "Final refit resource report: model=%s rows=%d cols=%d "
        "X_dtype_itemsize=%d X_bytes=%.1fMB available_ram=%.0fMB "
        "final_cpu_n_jobs=%d estimator_count=%s est_fit_memory=%.1fMB "
        "budget=%.0fMB device=%s",
        name,
        rows,
        cols,
        itemsize,
        x_bytes_mb,
        available_mb,
        n_jobs,
        n_trees if n_trees is not None else "n/a",
        estimate_mb if estimate_mb is not None else float("nan"),
        budget_mb,
        device,
    )

    # The hard-fail RAM guard applies only to CPU final fits of tree-bounded
    # estimators. A GPU fit computes on VRAM; a non-tree estimator (MLP) is not
    # tree-bounded and we do not hard-fail on it.
    if device != "cpu" or estimate_mb is None:
        return

    total_mb = estimate_mb + x_bytes_mb + 512.0  # baseline interpreter/loader

    if total_mb <= budget_mb:
        logger.info(
            "Final refit resource estimate %.1fMB within budget %.0fMB.",
            total_mb,
            budget_mb,
        )
        return

    # Over budget. If auto-downscale is enabled, reduce the estimator count.
    if getattr(config, "final_fit_auto_downscale", False):
        fit_trees = _downscale_for_budget(name, est, rows, y.shape[1], budget_mb, n_jobs)
        if fit_trees is not None and fit_trees < n_trees:
            if _downscale_estimator(est, fit_trees):
                new_estimate = estimate_tree_fit_memory_mb(
                    name, est, rows, y.shape[1], n_jobs=n_jobs
                )
                logger.warning(
                    "Final refit downscaled %s estimator_count %d -> %d "
                    "to fit %.0fMB budget (estimated %.1fMB).",
                    name,
                    n_trees,
                    fit_trees,
                    budget_mb,
                    new_estimate if new_estimate is not None else float("nan"),
                )
                return

    raise RuntimeError(
        "Selected NPS model '%s' cannot safely fit %d rows / %d outputs under "
        "final_fit_memory_budget_mb=%.0fMB. Estimated final-fit memory "
        "%.1fMB (X=%.1fMB). Reduced CPU parallelism to n_jobs=%d already. "
        "No rows were dropped and the model was not substituted. To proceed: "
        "raise final_fit_memory_budget_mb, enable "
        "final_fit_auto_downscale=True, or select a lighter model."
        % (name, rows, y.shape[1], budget_mb, total_mb, x_bytes_mb, n_jobs)
    )
