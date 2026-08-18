"""
AxiPulseAI – Optional GPU acceleration for the FINAL NPS training fit.

Design contract:
- GPU is used ONLY for the final selected-model full-data fit.
- CV / model selection always stays on CPU (no GPU memory allocated there).
- GPU is opt-in via ``Config.use_gpu`` (defaults to True) and is used only
  for the final full-data fit.
- If the GPU is unavailable (no driver / unsupported library build), training
  falls back to CPU automatically and never crashes because of it.
- Only CatBoost / XGBoost (and LightGBM when its build supports GPU) use GPU.
  ExtraTrees, RandomForest, HistGradientBoosting, GradientBoosting and MLP
  always stay on CPU.
- Conservative GPU settings suitable for a 6 GB card. Model complexity is
  never increased because a GPU is present.
"""

import ctypes
import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Model families that may use the GPU for the final fit. Everything not in
# this set is always trained on CPU.
_GPU_ELIGIBLE = {"CatBoost", "XGBoost", "LightGBM"}
_CPU_ONLY = {
    "ExtraTrees",
    "RandomForest",
    "HistGradientBoosting",
    "GradientBoosting",
    "MLP",
}

# Env override that forces the CPU path even when a GPU is present. Useful for
# CI machines and for verifying the fallback.
_DISABLE_ENV = "AXIPULSE_DISABLE_GPU"

_gpu_available_cache = None
_lgb_gpu_cache = None


def gpu_available() -> bool:
    """Return True when a usable CUDA GPU driver is present.

    Never raises. The result is cached for the lifetime of the process.
    Respects the ``AXIPULSE_DISABLE_GPU`` environment override.
    """
    global _gpu_available_cache

    if _gpu_available_cache is not None:
        return _gpu_available_cache

    if _env_disables_gpu():
        _gpu_available_cache = False
        return False

    _gpu_available_cache = _detect_driver()
    return _gpu_available_cache


def reset_cache():
    """Clear cached detection results (used by tests)."""
    global _gpu_available_cache, _lgb_gpu_cache
    _gpu_available_cache = None
    _lgb_gpu_cache = None


def _env_disables_gpu() -> bool:
    value = _getenv(_DISABLE_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "disable"}


def _getenv(key):
    import os

    return os.environ.get(key)


def _detect_driver() -> bool:
    # 1. Try loading the CUDA driver library directly (Linux/macOS).
    for lib in ("libcuda.so.1", "libcuda.dylib", "nvcuda.dll"):
        try:
            ctypes.CDLL(lib)
            return True
        except Exception:
            continue

    # 2. Fall back to nvidia-smi.
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False

    try:
        result = subprocess.run(
            [exe, "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def lightgbm_gpu_supported() -> bool:
    """Return True only when the installed LightGBM build demonstrably
    supports GPU training (the pip wheels are frequently CPU-only)."""
    global _lgb_gpu_cache

    if _lgb_gpu_cache is not None:
        return _lgb_gpu_cache

    _lgb_gpu_cache = _probe_lightgbm_gpu()
    return _lgb_gpu_cache


def _probe_lightgbm_gpu() -> bool:
    try:
        from lightgbm import LGBMRegressor
        import numpy as np
    except Exception:
        return False

    if LGBMRegressor is None:
        return False

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8)).astype(np.float32)
    y = rng.normal(size=64).astype(np.float32)

    try:
        model = LGBMRegressor(n_estimators=1, num_leaves=4, device="gpu")
        model.fit(X, y, verbose=False)
        return True
    except Exception:
        return False


def gpu_free_vram_mb() -> Optional[float]:
    """Return the free VRAM in MiB for the first GPU, or None if unknown.

    Uses ``nvidia-smi --query-gpu=memory.free``.  Returns None when the query
    fails so callers can treat the result conservatively (do not assume
    feasibility from an unknown value).
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None

    try:
        result = subprocess.run(
            [
                exe,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().splitlines()
        if not line:
            return None
        return float(line[0].strip())
    except Exception:
        return None


def gpu_final_fit_feasible(config) -> bool:
    """Return whether the GPU final fit is feasible under the resource policy.

    "GPU driver exists" is NOT sufficient.  When the config declares a
    ``gpu_min_free_vram_mb`` threshold, the current free VRAM must be at or
    above it.  If free VRAM cannot be determined, feasibility is rejected
    (conservative) rather than assumed — unless no threshold is configured.
    """
    threshold = float(getattr(config, "gpu_min_free_vram_mb", 0.0))
    if threshold <= 0.0:
        # No threshold configured: fall back to driver-present semantics.
        return True
    free = gpu_free_vram_mb()
    if free is None:
        # Cannot verify the GPU actually has headroom; do not claim feasible.
        logger.warning(
            "GPU final fit feasibility: unable to read free VRAM; "
            "rejecting GPU selection under gpu_min_free_vram_mb=%s.",
            threshold,
        )
        return False
    return free >= threshold


def select_final_fit_device(model_name: str, config) -> str:
    """Return ``"gpu"`` or ``"cpu"`` for the final full-data fit.

    Decision rules:
    - GPU disabled -> "cpu".
    - GPU driver unavailable -> "cpu".
    - Model not GPU-eligible (e.g. MLP / forests) -> "cpu".
    - LightGBM without a GPU-capable build -> "cpu".
    - Free VRAM below ``gpu_min_free_vram_mb`` (when configured) -> "cpu".
    - Otherwise -> "gpu".
    """
    use_gpu = bool(getattr(config, "use_gpu", False))

    if not use_gpu:
        return "cpu"

    if not gpu_available():
        return "cpu"

    if model_name not in _GPU_ELIGIBLE:
        return "cpu"

    if model_name == "LightGBM" and not lightgbm_gpu_supported():
        return "cpu"

    if not gpu_final_fit_feasible(config):
        return "cpu"

    return "gpu"


def apply_gpu_params(model, model_name: str, config) -> bool:
    """Configure the underlying estimator for GPU training in place.

    Returns True when the model was reconfigured for GPU, False when it stays
    on CPU. Conservative settings for a ~6 GB card are used; model complexity
    is left untouched.
    """
    if select_final_fit_device(model_name, config) != "gpu":
        return False

    try:
        if model_name == "CatBoost":
            # CatBoost multi-output is a native CatBoostRegressor, or is
            # wrapped in MultiOutputRegressor in the non-multi path.
            est = getattr(model, "estimator", model)
            est.set_params(task_type="GPU", devices="0")
            return True

        if model_name == "XGBoost":
            # Always wrapped in MultiOutputRegressor.
            est = model.estimator
            est.set_params(tree_method="hist", device="cuda", n_jobs=1)
            return True

        if model_name == "LightGBM":
            est = model.estimator
            est.set_params(device="gpu", gpu_use_dp=True, n_jobs=1)
            return True
    except Exception as exc:
        logger.warning(
            "GPU configuration failed for %s (%s); falling back to CPU.",
            model_name,
            exc,
        )
        return False

    return False


def gpu_memory_info() -> str:
    """Return a short, safe VRAM usage string (or empty string on failure)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ""

    try:
        result = subprocess.run(
            [
                exe,
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""

        line = result.stdout.strip().splitlines()
        if not line:
            return ""

        used, total, util = [part.strip() for part in line[0].split(",")]
        return f"{used} MiB / {total} MiB VRAM, util {util}%"
    except Exception:
        return ""
