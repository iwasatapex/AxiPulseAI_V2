"""Thin service layer for the GUI.

Every function here DELEGATES to the canonical V2 service / engine. The
GUI does not own any business logic. The only state owned here is the
"active model family" pointer, which is set through the canonical
``PredictorProvider`` so downstream predictions honor it.
"""
from __future__ import annotations

import datetime
import gc
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
from core.forecast_ai.models import (
    ForecastRequest,
    OperationType,
    ScenarioType,
)
from core.forecast_ai.prediction import PredictorProvider
from core.forecast_ai.prediction.model_selector import (
    MODELS_DIR,
    ModelPairError,
    list_model_families,
    list_training_files,
    validate_model_pair,
)
from core.forecast_ai.scenarios.registry import ScenarioRegistry

from . import contracts as ct
from .state import STATE

logger = logging.getLogger(__name__)


# Serialise access to the shared process-global PredictorProvider.  The
# canonical provider is a process-wide singleton, so per-request isolation
# is achieved by holding this lock from "set the family" through the
# provider-bound execution.  This prevents two Streamlit sessions from
# racing on the shared selector (one session cannot change the model a
# concurrent request is using).
_PROVIDER_LOCK = threading.RLock()


# =====================================================================
# Datasets
# =====================================================================

def list_datasets() -> List[Dict[str, Any]]:
    """List every file in ``training/`` (canonical V2 listing)."""
    files = list_training_files()
    out = []
    for f in files:
        try:
            stat = f.stat()
            size = stat.st_size
        except OSError:
            size = 0
        try:
            with f.open("rb") as fp:
                head = fp.read(2048)
            text = True
        except OSError:
            text = False
            head = b""
        ext = f.suffix.lstrip(".").lower() or "(none)"
        detected = _detect_type(f.name, head)
        out.append(
            {
                "name": f.name,
                "stem": f.stem,
                "path": str(f),
                "size_bytes": size,
                "modified": datetime.datetime.fromtimestamp(
                    f.stat().st_mtime
                ).isoformat()
                if f.exists()
                else None,
                "extension": ext,
                "type": detected,
                # Only advertise formats the canonical loader can actually
                # train/preview.  Everything else is listed but explicitly
                # marked as not trainable so the UI never offers it.
                "trainable": f.suffix.lower() in ct.SUPPORTED_DATASET_FORMATS,
            }
        )
    return out


def _detect_type(name: str, head: bytes) -> str:
    n = name.lower()
    if n.endswith(".csv"):
        return "csv"
    if n.endswith(".tsv"):
        return "tsv"
    if n.endswith(".json") or head.lstrip().startswith(b"{"):
        return "json"
    if n.endswith(".parquet"):
        return "parquet"
    if n.endswith(".xlsx") or n.endswith(".xls"):
        return "excel"
    if n.endswith(".pkl") or n.endswith(".joblib"):
        return "model"
    if n.endswith(".txt") or n.endswith(".log"):
        return "text"
    return "unknown"


def preview_dataset(name: str, n_rows: int = 5) -> Dict[str, Any]:
    files = {f.name: f for f in list_training_files()}
    if name not in files:
        raise FileNotFoundError(f"Training file not found: {name}")
    f = files[name]
    df = ct.load_dataset_sample(f, n_rows=n_rows)
    return {
        "columns": list(df.columns),
        "rows": df.head(n_rows).fillna("").astype(str).to_dict(orient="records"),
    }


# =====================================================================
# Training
# =====================================================================

def train_models(
    dataset_name: str,
    progress: Optional[List[str]] = None,
    progress_lock: Optional[threading.Lock] = None,
    status: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train OH + NPS from ONE dataset. Delegates to the existing trainers.

    Output: ``{MODELS_DIR}/{stem}_OH.pkl`` and
    ``{MODELS_DIR}/{stem}_NPS.pkl`` — exactly the V2 naming convention
    used by ``model_selector`` (absolute paths, independent of cwd).
    Re-training the same family overwrites the existing pair (no versioned
    duplicates), consistent with the existing CLI behavior.

    ``status`` (optional) is a thread-safe :class:`TrainingProgress` object
    that the GUI polls while training runs. When provided, it is forwarded to
    both trainers so live stage/model/fold progress is visible.

    NOTE: this does NOT activate the trained family.  The caller (the Train
    view, on the main Streamlit thread) must activate it via
    ``select_model_family`` so only the initiating session's selection is
    updated.
    """
    try:
        return _train_models_impl(
            dataset_name,
            progress=progress,
            progress_lock=progress_lock,
            status=status,
        )
    except Exception as exc:  # noqa: BLE001
        if status is not None:
            try:
                status.fail(str(exc))
            except Exception:  # pragma: no cover - advisory
                pass
        raise


def _train_models_impl(
    dataset_name: str,
    progress: Optional[List[str]] = None,
    progress_lock: Optional[threading.Lock] = None,
    status: Optional[Any] = None,
) -> Dict[str, Any]:
    from core.operation_health_predictor.predictor import (
        OperationalHealthPredictor,
    )
    from core.nps_predictor.predictor import NPSPredictor

    files = {f.name: f for f in list_training_files()}
    if dataset_name not in files:
        raise FileNotFoundError(f"Training file not found: {dataset_name}")
    selected = files[dataset_name]
    if selected.suffix.lower() not in ct.SUPPORTED_DATASET_FORMATS:
        raise ValueError(
            f"Dataset {selected.name!r} uses unsupported format "
            f"{selected.suffix or '(none)'}. Supported training formats: "
            + ", ".join(sorted(ct.SUPPORTED_DATASET_FORMATS))
        )
    family = selected.stem
    oh_out = MODELS_DIR / f"{family}{ct.MODEL_SUFFIX_OH}"
    nps_out = MODELS_DIR / f"{family}{ct.MODEL_SUFFIX_NPS}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if progress is not None and progress_lock is not None:
            with progress_lock:
                progress.append(msg)

    # ---- Validate dataset has both OH and NPS columns ----
    df_head = ct.load_dataset_sample(selected, n_rows=5)
    cols = set(df_head.columns)
    oh_required = {
        "actual_quality",
        "actual_competency",
        "actual_attendance",
        "actual_release_rate",
        "actual_transfer_rate",
    }
    nps_required = {"promoters", "passives", "detractors"}
    if not oh_required.issubset(cols):
        missing = oh_required - cols
        raise ValueError(
            f"Dataset missing OH-required columns: {sorted(missing)}"
        )
    if not nps_required.issubset(cols):
        missing = nps_required - cols
        raise ValueError(
            f"Dataset missing NPS-required columns: {sorted(missing)}"
        )

    # Release the tiny validation head before heavy training starts.
    del df_head

    if status is not None:
        try:
            status.set_kind("OH")
            status.set_stage("loading", message=f"Training dataset {selected.name}")
        except Exception:  # pragma: no cover - advisory
            pass

    _log(f"Training dataset : {selected.name}")
    _log(f"Model family     : {family}")
    _log(f"Output (OH)      : {oh_out}")
    _log(f"Output (NPS)     : {nps_out}")
    _log("Step 1/2: Training OH ...")
    oh = OperationalHealthPredictor()
    if status is not None:
        oh.train(str(selected), progress=status)
    else:
        oh.train(str(selected))
    if status is not None:
        try:
            status.set_stage("saving", message="Saving OH model")
        except Exception:  # pragma: no cover - advisory
            pass
    oh.save_model(str(oh_out))

    # Extract only small metadata/metrics before releasing OH.
    oh_metrics = _safe_metrics(oh)
    oh_algorithm = getattr(oh, "model_name", None)
    oh_features = len(getattr(oh, "feature_names", []) or [])

    _log("Step 1/2: OH complete")

    # CRITICAL: release OH's potentially huge DataFrame/model state
    # before NPS starts loading the same 1M-row dataset.
    del oh
    gc.collect()

    _log("OH memory released")
    _log("Step 2/2: Training NPS ...")

    if status is not None:
        try:
            status.set_kind("NPS")
            status.set_stage("loading", message="Training NPS model")
        except Exception:  # pragma: no cover - advisory
            pass

    nps = NPSPredictor()
    if status is not None:
        nps.train(str(selected), progress=status)
    else:
        nps.train(str(selected))
    if status is not None:
        try:
            status.set_stage("saving", message="Saving NPS model")
        except Exception:  # pragma: no cover - advisory
            pass
    nps.save_model(str(nps_out))

    # Extract only small metadata/metrics before releasing NPS.
    nps_metrics = _safe_metrics(nps)
    nps_algorithm = getattr(nps, "model_name", None)
    nps_features = len(getattr(nps, "feature_names", []) or [])

    _log("Step 2/2: NPS complete")

    # Release NPS training memory as well.
    del nps
    gc.collect()

    _log("NPS memory released")

    # The freshly trained family is a CANDIDATE, not production.  Training
    # writes only ``{family}_OH.pkl`` / ``{family}_NPS.pkl`` and must never
    # touch the canonical ``production_*`` artifacts.  Promotion into
    # production is an explicit, separate action (see ``promote_production``).
    _log(f"Candidate family '{family}' trained (OH+NPS written). "
         f"Promote explicitly to make it the production pair.")

    if status is not None:
        try:
            status.complete(
                model_name=nps_algorithm,
                rows=nps_features,
            )
        except Exception:  # pragma: no cover - advisory
            pass

    return {
        "family": family,
        "oh_path": str(oh_out),
        "nps_path": str(nps_out),
        "oh_metrics": oh_metrics,
        "nps_metrics": nps_metrics,
        "oh_algorithm": oh_algorithm,
        "nps_algorithm": nps_algorithm,
        "oh_features": oh_features,
        "nps_features": nps_features,
        "trained_at": datetime.datetime.now().isoformat(),
    }


def _safe_metrics(predictor: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    perf = getattr(predictor, "algorithm_performance", None)
    if isinstance(perf, dict) and perf:
        for k, v in perf.items():
            if isinstance(v, dict):
                out[k] = {kk: _coerce(vv) for kk, vv in v.items()}
            else:
                out[k] = _coerce(v)
    hist = getattr(predictor, "history_days", None)
    if hist is not None:
        out["history_days"] = int(hist)
    return out


def _coerce(v: Any) -> Any:
    try:
        if isinstance(v, (int, float, str, bool)):
            return v
        if hasattr(v, "item"):
            return v.item()
        return str(v)
    except Exception:
        return str(v)


def do_promote_production(family: str, models_dir=None) -> Dict[str, str]:
    """Explicitly promote a candidate family into the canonical production slot.

    The candidate pair is ``{family}_OH.pkl`` / ``{family}_NPS.pkl`` written by
    a prior training run.  This performs an atomic update of the production
    registry (copying the candidate bundles to ``production_OH.pkl`` /
    ``production_NPS.pkl`` and the legacy mirrors, then rewriting the integrity
    manifest).  It is only meant to be called from a deliberate UI action
    (e.g. a "Promote candidate" button).

    Raises :class:`FileNotFoundError` if the candidate pair has not been
    trained, or :class:`ModelPairError` if it is incomplete.
    """
    from core.forecast_ai.prediction.production_registry import register_production

    d = Path(models_dir) if models_dir else MODELS_DIR
    oh_candidate = d / f"{family}{ct.MODEL_SUFFIX_OH}"
    nps_candidate = d / f"{family}{ct.MODEL_SUFFIX_NPS}"
    if not oh_candidate.exists() or not nps_candidate.exists():
        raise FileNotFoundError(
            f"Candidate family '{family}' not found in {d}. Train a candidate "
            f"first, then promote explicitly."
        )

    registered = register_production(family, models_dir)
    logger.info("Promoted candidate family '%s' to production.", family)
    return registered


# =====================================================================
# Models
# =====================================================================

def list_models() -> List[Dict[str, Any]]:
    """List complete model families (OH + NPS pair)."""
    out = []
    for fam in list_model_families():
        try:
            oh_path, nps_path = validate_model_pair(fam)
            oh_info = _inspect_model(oh_path)
            nps_info = _inspect_model(nps_path)
            out.append(
                {
                    "family": fam,
                    "oh_path": str(oh_path),
                    "nps_path": str(nps_path),
                    "oh": oh_info,
                    "nps": nps_info,
                    "saved_at": _latest_mtime(oh_path, nps_path),
                    "active": STATE.get_active_family() == fam,
                }
            )
        except ModelPairError as exc:
            out.append({"family": fam, "error": str(exc)})
    return out


def _inspect_model(path: Path) -> Dict[str, Any]:
    import joblib

    try:
        data = joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive
        return {"path": str(path), "error": str(exc)}
    info: Dict[str, Any] = {
        "path": str(path),
        "model_name": data.get("model_name"),
        "algorithm": data.get("model_name"),
        "feature_count": len(data.get("feature_names") or []),
        "feature_names_sample": list(
            (data.get("feature_names") or [])[:8]
        ),
        "trained": bool(data.get("trained")),
        "history_days": data.get("history_days"),
        # NPS files store engine_version at the top level, but OH files store
        # it only inside ``metadata``. Fall back so both show a real version
        # instead of ``None`` for legacy OH models.
        "engine_version": data.get("engine_version")
        or (data.get("metadata") or {}).get("engine_version"),
    }
    meta = data.get("metadata")
    if isinstance(meta, dict):
        tr = meta.get("training_rows")
        if tr is not None:
            info["training_rows"] = _coerce(tr)
        dev = meta.get("device") or meta.get("hardware")
        if dev is not None:
            info["device"] = _coerce(dev)
        libs = meta.get("library_versions")
        if isinstance(libs, dict):
            info["device"] = info.get("device") or ("GPU" if any(
                "gpu" in str(libs.get(k, "")).lower() for k in libs
            ) else None)
        msd = meta.get("model_selection_diagnostics")
        if isinstance(msd, dict):
            info["model_selection_diagnostics"] = {
                k: (_coerce(v) if not isinstance(v, dict) else {kk: _coerce(vv) for kk, vv in v.items()})
                for k, v in msd.items()
            }
    alg = data.get("algorithm_performance")
    if isinstance(alg, dict):
        info["algorithm_performance"] = {
            k: _coerce(v) for k, v in alg.items()
        }
        # The model's own algorithm MAE (its algorithm_performance key).
        mae_key = data.get("model_name")
        if mae_key in alg:
            info["mae"] = _coerce(alg[mae_key])
    if isinstance(meta, dict) and meta:
        info["metadata"] = {k: _coerce(v) for k, v in meta.items()}
    tp = data.get("tuned_params")
    if isinstance(tp, dict) and tp:
        info["tuned_params"] = {k: _coerce(v) for k, v in tp.items()}
    return info


def _latest_mtime(*paths: Path) -> Optional[str]:
    times = []
    for p in paths:
        try:
            times.append(p.stat().st_mtime)
        except OSError:
            pass
    if not times:
        return None
    return datetime.datetime.fromtimestamp(max(times)).isoformat()


def select_model_family(family: Optional[str]) -> Dict[str, Any]:
    if family is None or family == "":
        STATE.set_active_family(None)
        return {"active_family": None}
    try:
        validate_model_pair(family)
    except ModelPairError as exc:
        raise ModelPairError(str(exc))
    STATE.set_active_family(family)
    return {"active_family": STATE.get_active_family()}


# =====================================================================
# System health / readiness
# =====================================================================

def system_health() -> Dict[str, Any]:
    """Lightweight readiness check (no expensive forecast/model inference).

    Verifies the dependencies the GUI actually needs:
      * the configured models directory exists,
      * at least one complete OH+NPS model family is present and the
        active family (if any) is a valid pair,
      * the baseline scenario is available.

    Returns an overall status of ``Ready`` / ``Degraded`` / ``Unavailable``.
    """
    checks: Dict[str, Any] = {}

    # --- models ---
    families = list_model_families()
    models_ok = bool(families)
    checks["models"] = {
        "status": "Ready" if models_ok else "Unavailable",
        "available_families": families,
    }

    # --- active family validity ---
    active = STATE.get_active_family()
    if not active:
        checks["active_model"] = {
            "status": "Degraded",
            "detail": "No active model family selected.",
        }
    else:
        try:
            validate_model_pair(active)
            checks["active_model"] = {
                "status": "Ready",
                "family": active,
            }
        except ModelPairError as exc:
            checks["active_model"] = {
                "status": "Degraded",
                "detail": f"Active family invalid: {exc}",
            }

    # --- scenarios ---
    scenario_ids = {s["id"] for s in list_scenarios()}
    checks["scenarios"] = {
        "status": "Ready" if ct.BASELINE_SCENARIO_ID in scenario_ids else "Degraded",
        "baseline_available": ct.BASELINE_SCENARIO_ID in scenario_ids,
    }

    # --- overall ---
    if not models_ok:
        overall = "Unavailable"
    else:
        degraded = any(
            checks[k].get("status") == "Degraded"
            for k in ("active_model", "scenarios")
        )
        overall = "Degraded" if degraded else "Ready"

    return {
        "status": overall,
        "checks": checks,
        "active_family": active,
        "available_families": families,
    }



# =====================================================================
# RAM SAFETY
# =====================================================================

def _ram_available_mb() -> int:
    """Return currently available system RAM in MB."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            values = {}
            for line in f:
                key, value = line.split(":", 1)
                if key in ("MemAvailable", "MemTotal"):
                    values[key] = int(value.strip().split()[0]) // 1024
            return values.get("MemAvailable", 0)
    except Exception:
        return 0


def _ram_guard(required_mb: int = 4096) -> None:
    """Prevent expensive operations when available RAM is dangerously low."""
    available = _ram_available_mb()

    if available and available < required_mb:
        raise RuntimeError(
            f"RAM safety guard stopped the operation. "
            f"Only {available:,} MB RAM is available; "
            f"at least {required_mb:,} MB is required. "
            f"Close other applications or wait for memory to be released."
        )


# =====================================================================
# Target State Engine + Reverse Optimizer
# =====================================================================

def _load_family_bundles(family: str) -> Tuple[Any, Any]:
    """Load the OH+NPS bundle dicts for an explicitly chosen model family.

    The pair is validated first (both files must exist).  Loading happens here
    at the service boundary so the chosen model is passed to the engine
    explicitly — the engine never silently falls back to another model.
    """
    import joblib

    oh_path, nps_path = validate_model_pair(family)
    return joblib.load(str(oh_path)), joblib.load(str(nps_path))


def find_target_state(targets: Dict[str, float],
                      family: Optional[str] = None,
                      total_candidates: int = 100000,
                      batch_size: int = 5000) -> Dict[str, Any]:
    """Delegate to the canonical TargetStateEngine.

    ``targets`` maps target keys (e.g. ``operational_health``, ``nps``,
    ``release``, ``transfer``, ``quality``, ``competency``, ``attendance``)
    to their desired values. The engine runs a batched reverse-optimization
    across the model council and returns a recommended state, consensus
    predictions, and leaderboards. ``total_candidates``/``batch_size`` are
    forwarded unchanged (the GUI defaults match the engine; callers may lower
    them for faster, lighter searches).

    ``family`` (optional) is the explicitly chosen model pair to run the
    council on.  When supplied it is validated, its bundles are loaded here
    and injected into the engine — the engine never silently falls back to a
    different model.  When omitted the engine uses its legacy model files.

    Targets are validated against the canonical KPI hard bounds at the
    service boundary before reaching the engine.
    """
    _validate_targets(targets)

    # Never start a large reverse search when the machine is already
    # close to exhausting RAM.
    _ram_guard(required_mb=4096)

    from core.target_state_engine.engine import TargetStateEngine

    if family:
        oh_bundle, nps_bundle = _load_family_bundles(family)
        engine = TargetStateEngine(oh_bundle=oh_bundle, nps_bundle=nps_bundle)
    else:
        engine = TargetStateEngine()
    result = engine.find_target_state(
        targets,
        total_candidates=total_candidates,
        batch_size=batch_size,
    )
    if not isinstance(result, dict):
        raise ValueError("TargetStateEngine returned an unexpected result type")
    result = dict(result)
    result["active_family"] = family
    return result


# Map engine target keys -> canonical KPI contract keys for validation.
_TARGET_KEY_TO_KPI = {
    "operational_health": "operations_health",
    "oh": "operations_health",
    "nps": "nps",
    "release": "release",
    "transfer": "transfer",
    "quality": "quality",
    "competency": "competency",
    "attendance": "attendance",
}


def _validate_targets(targets: Dict[str, float]) -> None:
    """Validate each target key/value against the canonical KPI bounds.

    Raises ``ValueError`` naming the offending target(s).  Unknown keys are
    left untouched (forwarded to the engine as-is) so engine-supported
    extras are not dropped.
    """
    if not isinstance(targets, dict) or not targets:
        raise ValueError("At least one target is required.")
    errors: list[str] = []
    for key, raw in targets.items():
        kpi_key = _TARGET_KEY_TO_KPI.get(str(key).lower())
        if kpi_key is None or kpi_key not in ct.KPI:
            continue
        cfg = ct.KPI[kpi_key]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"Target {key!r} must be numeric, got {raw!r}")
            continue
        if cfg["min"] is not None and not (cfg["min"] <= value <= cfg["max"]):
            errors.append(
                f"Target {key!r} must be within [{cfg['min']:g}, {cfg['max']:g}]"
            )
    if errors:
        raise ValueError("Invalid target(s): " + "; ".join(errors))


_REVERSE_TARGET_KEY = {"OH": "operational_health", "NPS": "nps"}
_REVERSE_BOUNDS = {"OH": (ct.OH_MIN, ct.OH_MAX), "NPS": (ct.NPS_MIN, ct.NPS_MAX)}


def reverse_optimize(metric: str, target: float,
                     family: Optional[str] = None) -> Dict[str, Any]:
    """Reverse-optimise the KPIs that drive a single target metric.

    ``metric`` is either ``"OH"`` (Operational Health) or ``"NPS"``. This
    delegates to the canonical TargetStateEngine (the one and only reverse
    optimizer in the V2 engine) with a single target, and returns the best
    KPI combination found plus the achieved consensus value and distance.

    ``family`` (optional) is the explicitly chosen model pair to run the
    council on — forwarded to :func:`find_target_state`.

    The target is validated against the canonical metric range (OH 0-100,
    NPS -100..100) before reaching the engine.
    """
    if metric not in _REVERSE_TARGET_KEY:
        raise ValueError(f"Unknown reverse-optimisation metric: {metric}")

    lo, hi = _REVERSE_BOUNDS[metric]
    try:
        value = float(target)
    except (TypeError, ValueError):
        raise ValueError(f"Target {metric} must be numeric, got {target!r}")
    if not (lo <= value <= hi):
        raise ValueError(
            f"Target {metric} must be within [{lo:g}, {hi:g}], got {value:g}"
        )

    key = _REVERSE_TARGET_KEY[metric]
    # Single-metric interactive search: fewer candidates than the full target
    # state sweep keeps the reverse optimizer responsive while still covering
    # a wide KPI space.
    result = find_target_state(
        {key: value}, family=family, total_candidates=20000
    )

    recommended = result.get("recommended_state") or {}
    consensus = result.get("consensus") or {}
    predicted = consensus.get("oh") if metric == "OH" else consensus.get("nps")

    return {
        "metric": metric,
        "target": value,
        "found": bool(recommended),
        "distance": result.get("distance"),
        "predicted": predicted,
        "recommended_state": recommended,
        "consensus": consensus,
        "leaderboards": result.get("leaderboards") or {},
        "active_family": family,
    }


# =====================================================================
# Predict
# =====================================================================

def predict(
    state: Dict[str, Any], family: Optional[str] = None
) -> Dict[str, Any]:
    """Run a direct prediction via the canonical provider.

    ``family`` MUST be the active family. If not supplied, the currently
    active family is used. An explicit family is honored only if it is
    complete (validates the pair) and then becomes the active family.

    The canonical provider is process-global, so the family is activated
    and the prediction run while holding ``_PROVIDER_LOCK``.  This gives
    per-request isolation: two sessions can never race on the shared
    provider selector (session A's prediction always uses session A's
    family).
    """
    if family:
        select_model_family(family)
    # Use the explicit family (the user's selection) as the source of truth
    # for provider activation; fall back to the session's active family only
    # when none was passed explicitly. This avoids a set-then-re-read race on
    # the shared session store between selecting and activating a family.
    active = family if family else STATE.get_active_family()
    if not active:
        raise ModelPairError(
            "No active model family. Select a model on the Models page first."
        )
    # Service-boundary validation: out-of-range KPIs (release < 50,
    # transfer > 20, ...) are rejected here, not only by widgets.
    ct.validate_state(state)

    from core.forecast_ai.prediction.pipeline import (
        ProductionPredictionPipeline,
    )

    with _PROVIDER_LOCK:
        PredictorProvider.set_model_family(active)
        pipeline = ProductionPredictionPipeline()
        result = pipeline.run(state=dict(state))

    payload = _prediction_to_dict(result)
    payload["_timestamp"] = datetime.datetime.now().isoformat()
    payload["active_family"] = active
    STATE.set_last_prediction(payload)
    return payload


def _prediction_to_dict(result: Any) -> Dict[str, Any]:
    raw = getattr(result, "prediction", result)
    raw_obj = getattr(raw, "raw", raw)
    out: Dict[str, Any] = {
        "operational_health": None,
        "nps": None,
        "promoters": None,
        "passives": None,
        "detractors": None,
        "distribution": {},
        "score_counts": {},
        "bayesian_score_distribution": {},
        "ensemble_details": {},
    }
    # ProductionPredictionResult attaches envelopes for OH + NPS.
    oh_env = getattr(raw, "operations_health", None)
    nps_env = getattr(raw, "nps", None)
    if oh_env is not None:
        out["operational_health"] = _coerce(getattr(oh_env, "prediction", None))
        oh_prob = getattr(oh_env, "probabilistic", None)
        if oh_prob is not None:
            out["oh_confidence"] = _coerce(getattr(oh_prob, "confidence", None))
            out["oh_lower"] = _coerce(getattr(oh_prob, "likely_range_lower", None))
            out["oh_upper"] = _coerce(getattr(oh_prob, "likely_range_upper", None))
    if nps_env is not None:
        out["nps"] = _coerce(getattr(nps_env, "prediction", None))
        nps_prob = getattr(nps_env, "probabilistic", None)
        if nps_prob is not None:
            out["nps_confidence"] = _coerce(getattr(nps_prob, "confidence", None))
            out["nps_lower"] = _coerce(getattr(nps_prob, "likely_range_lower", None))
            out["nps_upper"] = _coerce(getattr(nps_prob, "likely_range_upper", None))

    # Native NPS 0..10 distribution preserved at the production boundary.
    out["bayesian_score_distribution"] = dict(
        getattr(raw, "bayesian_score_distribution", {}) or {}
    )
    out["score_counts"] = dict(getattr(raw, "score_counts", {}) or {})

    # Propagate any prediction errors reported by the underlying service.
    raw_errors = getattr(raw_obj, "errors", None)
    if raw_errors:
        out["errors"] = [str(e) for e in raw_errors]

    # Pull values out of the underlying PredictionResult dataclass.
    for attr in (
        "operational_health",
        "nps",
        "promoters",
        "passives",
        "detractors",
        "distribution",
        "ensemble_details",
    ):
        if hasattr(raw_obj, attr):
            val = getattr(raw_obj, attr)
            if val is None:
                continue
            if attr in ("distribution", "ensemble_details") and isinstance(val, dict):
                out[attr] = {k: _coerce(v) for k, v in val.items()}
            else:
                out[attr] = _coerce(val)
    return out


# =====================================================================
# Forecast
# =====================================================================

def list_scenarios(include_disabled: bool = False) -> List[Dict[str, Any]]:
    """Built-in scenarios registered with ForecastAI.

    Only **enabled** scenarios are returned by default so a disabled
    scenario can never be offered for selection/execution.  ``baseline``
    (the default no-op) is appended exactly once — it is never registered
    in ``ScenarioRegistry``, and it is deduplicated against any existing
    ``baseline`` entry so no duplicate baseline ever appears.
    """
    out: Dict[str, Dict[str, Any]] = {}

    def _add(sid, name, description, enabled=None, priority=None):
        if sid not in out:
            entry = {"id": sid, "name": name, "description": description}
            if enabled is not None:
                entry["enabled"] = enabled
            if priority is not None:
                entry["priority"] = priority
            out[sid] = entry

    for s in ScenarioRegistry.list():
        sid = getattr(s, "id", None)
        if sid is None:
            continue
        enabled = bool(getattr(s, "enabled", True))
        if not include_disabled and not enabled:
            continue  # disabled scenarios are not selectable/executable
        _add(
            sid,
            getattr(s, "name", None),
            getattr(s, "description", None),
            enabled=enabled,
            priority=getattr(s, "priority", None),
        )

    # Baseline is the default no-op. Append exactly once (dedup).
    _add(
        ct.BASELINE_SCENARIO_ID,
        "Baseline",
        "No-op default scenario.",
        enabled=True,
        priority=0,
    )
    return list(out.values())


def forecast(
    state: Dict[str, Any],
    horizon: int,
    scenario: Optional[str] = None,
    family: Optional[str] = None,
    update_state: bool = True,
    target_oh: Optional[float] = None,
    target_nps: Optional[float] = None,
) -> Dict[str, Any]:
    """Run ForecastOrchestrator.execute and return a JSON-friendly payload.

    ``update_state`` (default True) controls whether the result is recorded as
    the session's "latest forecast" (used by the dashboard). Scenario
    comparison passes ``False`` so a comparison run does not replace the
    single-scenario forecast the user last ran.

    ``target_oh`` / ``target_nps`` (optional) enable the Forecast AI
    recommendation engine and target-driven ADIE detail (Monte Carlo success
    rate, per-metric target probabilities, KPI-specific recommendations).
    """
    if family:
        select_model_family(family)
    # Use the explicit family for provider activation; fall back to the
    # session's active family only when none was passed explicitly (avoids
    # a set-then-re-read race on the shared session store).
    active = family if family else STATE.get_active_family()
    if not active:
        raise ModelPairError(
            "No active model family. Select a model on the Models page first."
        )
    if horizon < 1:
        raise ValueError("Horizon must be at least 1 day.")
    # Service-boundary validation before any engine call.
    ct.validate_state(state)
    # Disabled scenarios must never be passed to the engine.
    _ensure_enabled_scenario(scenario)

    parameters = {"state": dict(state or {})}
    if target_oh is not None:
        parameters["target_oh"] = float(target_oh)
    if target_nps is not None:
        parameters["target_nps"] = float(target_nps)

    req = ForecastRequest(
        operation=OperationType.FORECAST,
        scenario=scenario or ScenarioType.BASELINE,
        horizon=horizon,
        parameters=parameters,
    )

    with _PROVIDER_LOCK:
        # Activate the session's family on the shared provider under the
        # lock so concurrent sessions cannot race on the selector.
        PredictorProvider.set_model_family(active)
        orchestrator = ForecastOrchestrator()
        response = orchestrator.execute(req)

    payload = _dataclass_to_dict(response.payload) if response.payload else {}
    timeline = payload.get("timeline", []) or []

    # The orchestrator does NOT emit top-level risk/sensitivity/confidence/
    # trend/recommendations/strategy keys.  They live nested inside
    # ``decision_intelligence.package``.  Reading the stale top-level keys
    # would silently show empty sections even though real data exists.
    decision = payload.get("decision_intelligence", {}) or {}
    if isinstance(decision, dict):
        dpackage = decision.get("package", {}) or {}
        if not isinstance(dpackage, dict):
            dpackage = {}
    else:
        dpackage = {}
    prob = dpackage.get("probabilistic", {}) or {}

    forecast_payload = {
        "success": bool(getattr(response, "success", False)),
        "horizon": horizon,
        "scenario": scenario or ct.BASELINE_SCENARIO_ID,
        "active_family": active,
        "engine": getattr(response, "engine", "ForecastOrchestrator"),
        "warnings": list(getattr(response, "warnings", []) or []),
        "errors": list(getattr(response, "errors", []) or []),
        "metadata": _dataclass_to_dict(getattr(response, "metadata", {}) or {}),
        "timeline": timeline,
        "summary": payload.get("summary", {}) or {},
        # Decision-layer outputs live in the ADIE V3 package; surface them at
        # the top level for the forecast view.  Risk/confidence at the day
        # level remain on each timeline row.
        "risk": prob.get("risk", {}) or {},
        "sensitivity": dpackage.get("sensitivity", {}) or {},
        "confidence": prob.get("confidence", {}) or {},
        "trend": dpackage.get("trends", {}) or {},
        "recommendations": dpackage.get("recommendations", {}) or {},
        "strategy": dpackage.get("strategies", {}) or {},
        "agreement": dpackage.get("agreement", {}) or {},
        # Expose the orchestrator's pre-built decision_intelligence so
        # adie_decision can reuse it (exactly-one-Monte-Carlo invariant).
        "decision_intelligence": decision,
        "_timestamp": datetime.datetime.now().isoformat(),
    }
    if update_state:
        STATE.set_last_forecast(forecast_payload)
    return forecast_payload


def _ensure_enabled_scenario(scenario: Optional[str]) -> None:
    """Reject a scenario that is disabled or unknown before engine execution."""
    if not scenario or scenario == ct.BASELINE_SCENARIO_ID:
        return  # baseline is the default no-op — always allowed.
    scenario_obj = ScenarioRegistry.get(scenario)
    if scenario_obj is None:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Select one of the available "
            f"scenarios."
        )
    if not getattr(scenario_obj, "enabled", True):
        raise ValueError(
            f"Scenario {scenario!r} is disabled and cannot be executed."
        )


def _dataclass_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _dataclass_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dataclass_to_dict(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            k: _dataclass_to_dict(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return str(value)


# =====================================================================
# ADIE V3
# =====================================================================

def adie_decision(
    state: Dict[str, Any],
    horizon: int,
    scenario: Optional[str] = None,
    family: Optional[str] = None,
    target_oh: Optional[float] = None,
    target_nps: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the ADIE V3 decision pipeline, composed with ForecastAI outputs.

    This does NOT duplicate any decision math. The forecast output is run
    first (which already executes sensitivity / risk / trend / etc.),
    then the V3 service composes the canonical decision package and
    folds the ForecastAI outputs into the same payload.

    ``target_oh`` / ``target_nps`` (optional) enable recommendations and the
    target-driven ADIE detail (Monte Carlo success, target probabilities).
    """
    if family:
        select_model_family(family)
    if not (family or STATE.get_active_family()):
        raise ModelPairError(
            "No active model family. Select a model on the Models page first."
        )

    forecast_payload = forecast(
        state=state, horizon=horizon, scenario=scenario, family=family,
        target_oh=target_oh, target_nps=target_nps,
    )
    if not forecast_payload.get("success"):
        return {
            "success": False,
            "decision": None,
            "forecast": forecast_payload,
            "raw": None,
            "errors": forecast_payload.get("errors", []),
            "warnings": forecast_payload.get("warnings", []),
        }

    from datetime import datetime as _dt

    # --- Reuse the orchestrator's pre-built ADIE package (exactly one MC). ---
    # ``forecast_payload['decision_intelligence']`` is the v3_decision dict
    # produced by ``ForecastOrchestrator._build_adie_v3_decision()`` — the
    # one and only Bayesian + Monte Carlo execution in the V2 forecast ->
    # ADIE direction. Reusing it here guarantees the GUI path never triggers
    # a second probabilistic pass and never calls ADIE predictors.
    v3_decision = forecast_payload.get("decision_intelligence") or {}
    package = v3_decision.get("package") or {}
    if v3_decision.get("status") != "success" or not package:
        # ADIE V3 handoff was skipped/failed (advisory-only fail-soft).
        # Report it explicitly instead of fabricating a decision.
        cutoff_ts = _dt.utcnow().isoformat()
        note = str(
            v3_decision.get("reason")
            or v3_decision.get("error")
            or "ADIE V3 decision not produced by forecast"
        )
        payload = {
            "success": True,
            "decision": {},
            "forecast": forecast_payload,
            "decision_intelligence": dict(v3_decision),
            "details": {},
            "raw": dict(v3_decision),
            "errors": [],
            "warnings": [note],
            "_timestamp": cutoff_ts,
        }
        STATE.set_last_adie(payload)
        return payload

    probabilistic = package.get("probabilistic") or {}
    details = package.get("details") or {}

    cutoff_ts = forecast_payload.get("_timestamp") or _dt.utcnow().isoformat()

    payload = {
        "success": True,
        "decision": package,
        "forecast": {
            "horizon": forecast_payload.get("horizon"),
            "scenario": forecast_payload.get("scenario"),
            "timeline": forecast_payload.get("timeline"),
            "summary": forecast_payload.get("summary"),
            "risk": forecast_payload.get("risk"),
            "sensitivity": forecast_payload.get("sensitivity"),
            "confidence": forecast_payload.get("confidence"),
            "trend": forecast_payload.get("trend"),
            "recommendations": package.get("recommendations") or forecast_payload.get("recommendations"),
            "strategy": package.get("strategies") or forecast_payload.get("strategy"),
        },
        "decision_intelligence": {
            "package": package,
            "probabilistic": probabilistic,
            "details": details,
            "recommendations": package.get("recommendations") or forecast_payload.get("recommendations"),
            "strategy": package.get("strategies") or forecast_payload.get("strategy"),
            "trend": package.get("trends") or forecast_payload.get("trend"),
            "sensitivity": package.get("sensitivity") or forecast_payload.get("sensitivity"),
        },
        "raw": package,
        "errors": [],
        "warnings": [],
        "_timestamp": cutoff_ts,
    }
    STATE.set_last_adie(payload)
    return payload
