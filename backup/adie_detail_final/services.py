"""Thin service layer for the GUI.

Every function here DELEGATES to the canonical V2 service / engine. The
GUI does not own any business logic. The only state owned here is the
"active model family" pointer, which is set through the canonical
``PredictorProvider`` so downstream predictions honor it.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
from core.forecast_ai.models import (
    ForecastRequest,
    OperationType,
    ScenarioType,
)
from core.forecast_ai.prediction import PredictorProvider
from core.forecast_ai.prediction.model_selector import (
    ModelPairError,
    list_model_families,
    list_training_files,
    validate_model_pair,
)
from core.forecast_ai.scenarios.registry import ScenarioRegistry

from .state import STATE

logger = logging.getLogger(__name__)


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
    if f.suffix.lower() == ".csv":
        df = pd.read_csv(f, nrows=n_rows)
    elif f.suffix.lower() in (".tsv",):
        df = pd.read_csv(f, sep="\t", nrows=n_rows)
    elif f.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(f, nrows=n_rows)
    else:
        raise ValueError(f"Cannot preview file of type {f.suffix}")
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
) -> Dict[str, Any]:
    """Train OH + NPS from ONE dataset. Delegates to the existing trainers.

    Output: ``models/{stem}_OH.pkl`` and ``models/{stem}_NPS.pkl`` —
    exactly the V2 naming convention used by ``model_selector``.
    Re-training the same family overwrites the existing pair (no versioned
    duplicates), consistent with the existing CLI behavior.
    """
    from core.operation_health_predictor.predictor import (
        OperationalHealthPredictor,
    )
    from core.nps_predictor.predictor import NPSPredictor

    files = {f.name: f for f in list_training_files()}
    if dataset_name not in files:
        raise FileNotFoundError(f"Training file not found: {dataset_name}")
    selected = files[dataset_name]
    family = selected.stem

    def _log(msg: str) -> None:
        if progress is not None and progress_lock is not None:
            with progress_lock:
                progress.append(msg)

    # ---- Validate dataset has both OH and NPS columns ----
    df_head = pd.read_csv(selected, nrows=5)
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

    _log(f"Training dataset : {selected.name}")
    _log(f"Model family     : {family}")
    _log(f"Output (OH)      : models/{family}_OH.pkl")
    _log(f"Output (NPS)     : models/{family}_NPS.pkl")
    _log("Step 1/2: Training OH ...")
    oh = OperationalHealthPredictor()
    oh.train(str(selected))
    oh.save_model(f"models/{family}_OH.pkl")
    _log("Step 1/2: OH complete")

    _log("Step 2/2: Training NPS ...")
    nps = NPSPredictor()
    nps.train(str(selected))
    nps.save_model(f"models/{family}_NPS.pkl")
    _log("Step 2/2: NPS complete")

    # Capture metrics where available, never fabricated.
    oh_metrics = _safe_metrics(oh)
    nps_metrics = _safe_metrics(nps)

    # Auto-activate the freshly trained family.
    STATE.set_active_family(family)

    return {
        "family": family,
        "oh_path": f"models/{family}_OH.pkl",
        "nps_path": f"models/{family}_NPS.pkl",
        "oh_metrics": oh_metrics,
        "nps_metrics": nps_metrics,
        "oh_algorithm": getattr(oh, "model_name", None),
        "nps_algorithm": getattr(nps, "model_name", None),
        "oh_features": len(getattr(oh, "feature_names", []) or []),
        "nps_features": len(getattr(nps, "feature_names", []) or []),
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
        "feature_count": len(data.get("feature_names") or []),
        "feature_names_sample": list(
            (data.get("feature_names") or [])[:8]
        ),
        "trained": bool(data.get("trained")),
        "engine_version": data.get("engine_version"),
    }
    alg = data.get("algorithm_performance")
    if isinstance(alg, dict):
        info["algorithm_performance"] = {
            k: _coerce(v) for k, v in alg.items()
        }
    meta = data.get("metadata")
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
# Predict
# =====================================================================

def predict(
    state: Dict[str, Any], family: Optional[str] = None
) -> Dict[str, Any]:
    """Run a direct prediction via the canonical provider.

    ``family`` MUST be the active family. If not supplied, the currently
    active family is used. An explicit family is honored only if it is
    complete (validates the pair) and then becomes the active family.
    """
    if family:
        select_model_family(family)
    active = STATE.get_active_family()
    if not active:
        raise ModelPairError(
            "No active model family. Select a model on the Models page first."
        )

    from core.forecast_ai.prediction.pipeline import (
        ProductionPredictionPipeline,
    )

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

def list_scenarios() -> List[Dict[str, Any]]:
    """Built-in scenarios registered with ForecastAI."""
    out = []
    for s in ScenarioRegistry.list():
        out.append(
            {
                "id": getattr(s, "id", None),
                "name": getattr(s, "name", None),
                "description": getattr(s, "description", None),
                "priority": getattr(s, "priority", None),
                "enabled": getattr(s, "enabled", None),
            }
        )
    out.append({"id": "baseline", "name": "Baseline", "description": "No-op default scenario."})
    return out


def forecast(
    state: Dict[str, Any],
    horizon: int,
    scenario: Optional[str] = None,
    family: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ForecastOrchestrator.execute and return a JSON-friendly payload."""
    if family:
        select_model_family(family)
    active = STATE.get_active_family()
    if not active:
        raise ModelPairError(
            "No active model family. Select a model on the Models page first."
        )
    if horizon < 1:
        raise ValueError("Horizon must be at least 1 day.")

    params = dict(state or {})
    params.setdefault("state", dict(state or {}))
    if scenario and scenario != "baseline":
        params["scenario_id"] = scenario

    req = ForecastRequest(
        operation=OperationType.FORECAST,
        scenario=scenario or ScenarioType.BASELINE,
        horizon=horizon,
        parameters={"state": dict(state or {})},
    )

    orchestrator = ForecastOrchestrator()
    response = orchestrator.execute(req)

    payload = _dataclass_to_dict(response.payload) if response.payload else {}
    timeline = payload.get("timeline", []) or []
    forecast_payload = {
        "success": bool(getattr(response, "success", False)),
        "horizon": horizon,
        "scenario": scenario or "baseline",
        "active_family": active,
        "engine": getattr(response, "engine", "ForecastOrchestrator"),
        "warnings": list(getattr(response, "warnings", []) or []),
        "errors": list(getattr(response, "errors", []) or []),
        "metadata": _dataclass_to_dict(getattr(response, "metadata", {}) or {}),
        "timeline": timeline,
        "summary": payload.get("summary", {}) or {},
        "risk": payload.get("risk", {}) or {},
        "sensitivity": payload.get("sensitivity", {}) or {},
        "confidence": payload.get("confidence", {}) or {},
        "trend": payload.get("trend", {}) or {},
        "recommendations": payload.get("recommendations", {}) or {},
        "strategy": payload.get("strategy", {}) or {},
        "_timestamp": datetime.datetime.now().isoformat(),
    }
    STATE.set_last_forecast(forecast_payload)
    return forecast_payload


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
) -> Dict[str, Any]:
    """Run the ADIE V3 decision pipeline, composed with ForecastAI outputs.

    This does NOT duplicate any decision math. The forecast output is run
    first (which already executes sensitivity / risk / trend / etc.),
    then the V3 service composes the canonical decision package and
    folds the ForecastAI outputs into the same payload.
    """
    if family:
        select_model_family(family)
    if not STATE.get_active_family():
        raise ModelPairError(
            "No active model family. Select a model on the Models page first."
        )

    forecast_payload = forecast(
        state=state, horizon=horizon, scenario=scenario
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

    from api.services.adie_v3_service import ADIEV3Service
    from datetime import datetime as _dt

    adie = ADIEV3Service()

    timeline = forecast_payload.get("timeline", [])
    observations = []
    for day in timeline:
        oh = day.get("operations_health") or day.get("operational_health")
        nps = day.get("nps")
        if isinstance(oh, (int, float)):
            observations.append(float(oh))
        if isinstance(nps, (int, float)):
            observations.append(float(nps))
    if not observations:
        observations = [
            float(state.get("operational_health") or 0.0),
        ]
    baseline = float(observations[0])
    observations = observations[1:] or observations

    scenarios = []
    # Build per-day scenarios from timeline data so ADIE gets real OH/NPS evidence.
    # The forecast timeline contains observed state at cutoff time.
    timeline = forecast_payload.get("timeline", []) or []
    if timeline:
        for idx, day in enumerate(timeline):
            oh_val = day.get("operations_health") or day.get("operational_health")
            nps_val = day.get("nps")
            if _finite(float(oh_val)) if isinstance(oh_val, (int, float)) else _finite(day.get("operations_health", 0)):
                oh = float(oh_val)
            else:
                oh = 0.0
            if _finite(float(nps_val)) if isinstance(nps_val, (int, float)) else _finite(day.get("nps", 0)):
                nps = float(nps_val)
            else:
                nps = 0.0
            scenarios.append({
                "id": f"day_{idx}",
                "name": f"Day {idx + 1}",
                "baseline": idx == 0,
                "operations_health": oh,
                "nps": nps,
                "confidence": day.get("confidence", 0.5),
                "probability": day.get("probability", 0.5),
                "delta_oh": day.get("delta_oh"),
                "risk_severity": day.get("risk_severity"),
            })
    if not scenarios:
        # Fallback: generic scenario
        if scenario and scenario != "baseline":
            scenarios.append({"id": scenario, "name": scenario, "modifiers": []})
        else:
            scenarios.append({"id": "baseline", "name": "Baseline", "modifiers": []})

    # Build recommendation/strategy/sensitivity/trend payloads if present.
    recommendation = forecast_payload.get("recommendations") or {}
    strategy = forecast_payload.get("strategy") or {}
    trend = forecast_payload.get("trend") or {}
    sensitivity = forecast_payload.get("sensitivity") or {}
    risk = forecast_payload.get("risk") or {}
    confidence = forecast_payload.get("confidence") or {}

        # --- Canonical cutoff timestamp (known at forecast time, not later). ---
    # Capture ONE timestamp and reuse it for every value that is supposed to be
    # "known at cutoff" so microsecond drift between separate ``now()`` calls
    # cannot trigger the temporal contract's "input after cutoff" check.
    cutoff_ts = _dt.utcnow().isoformat()

    package = adie.compose_decision(
        scenarios=scenarios or [{"id": "baseline", "name": "Baseline", "modifiers": []}],
        observations=observations,
        baseline=baseline,
        cutoff=cutoff_ts,
        metadata={"provenance": cutoff_ts},
        recommendation_output=recommendation,
        strategy_output=strategy,
        trend_output=trend,
        sensitivity_output=sensitivity,
        agreement={
            "risk": risk,
            "confidence": confidence,
        },
        observed=baseline,
        observed_metrics=["operational_health", "nps"],
        horizon=horizon,
    )

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
            "recommendations": forecast_payload.get("recommendations"),
            "strategy": forecast_payload.get("strategy"),
        },
        "decision_intelligence": {
            "package": package,
            "recommendations": forecast_payload.get("recommendations"),
            "strategy": forecast_payload.get("strategy"),
            "trend": forecast_payload.get("trend"),
            "sensitivity": forecast_payload.get("sensitivity"),
        },
                "raw": package,
        "errors": [],
        "warnings": [],
        "_timestamp": cutoff_ts,
    }
    STATE.set_last_adie(payload)
    return payload
