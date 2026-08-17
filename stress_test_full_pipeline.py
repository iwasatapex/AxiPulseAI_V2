#!/usr/bin/env python3
"""
AxiPulseAI V2 — FULL PIPELINE STRESS TEST

READ-ONLY:
- does not modify source code
- does not modify model artifacts
- does not retrain
- writes only a timestamped result .txt file

Run:
    cd /home/amteur/Documents/AxiPulseAI_V2
    source venv/bin/activate
    python stress_test_full_pipeline.py
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path("/home/amteur/Documents/AxiPulseAI_V2")
RESULT_DIR = PROJECT / "test_results"

sys.path.insert(0, str(PROJECT))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(value: Any) -> Any:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return value


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return json_safe(value.dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return json_safe(vars(value))
        except Exception:
            pass

    return str(value)


def compact(value: Any, max_len: int = 5000) -> str:
    text = json.dumps(json_safe(value), indent=2, ensure_ascii=False, default=str)
    if len(text) > max_len:
        return text[:max_len] + "\n...TRUNCATED..."
    return text


def finite_scan(value: Any, path: str = "root") -> list[str]:
    errors: list[str] = []

    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{path}: non-finite value {value}")
        return errors

    if isinstance(value, dict):
        for k, v in value.items():
            errors.extend(finite_scan(v, f"{path}.{k}"))
        return errors

    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            errors.extend(finite_scan(v, f"{path}[{i}]"))

    return errors


def get_value(obj: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            return getattr(obj, key)
    return default


def section(title: str) -> str:
    return (
        "\n"
        + "=" * 80
        + "\n"
        + title
        + "\n"
        + "=" * 80
        + "\n"
    )


# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

def load_components():
    from core.forecast_ai.prediction.model_selector import (
        list_model_families,
        validate_model_pair,
    )
    from core.forecast_ai.prediction.provider import PredictorProvider
    from core.forecast_ai.prediction.service import PredictionService
    from core.forecast_ai.prediction.models import PredictionRequest
    from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
    from core.forecast_ai.models import ForecastRequest

    return {
        "list_model_families": list_model_families,
        "validate_model_pair": validate_model_pair,
        "PredictorProvider": PredictorProvider,
        "PredictionService": PredictionService,
        "PredictionRequest": PredictionRequest,
        "ForecastOrchestrator": ForecastOrchestrator,
        "ForecastRequest": ForecastRequest,
    }


# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------

def choose_model_family(list_model_families, validate_model_pair) -> str:
    families = list_model_families()

    if not families:
        raise RuntimeError(
            "No complete OH+NPS model families found in V2/models/"
        )

    print("\nAVAILABLE MODEL FAMILIES")
    print("-" * 40)

    for i, family in enumerate(families, 1):
        print(f"{i}. {family}")

    while True:
        raw = input("\nSelect model family number: ").strip()

        try:
            idx = int(raw)
            if 1 <= idx <= len(families):
                family = families[idx - 1]
                validate_model_pair(family)
                return family
        except Exception as exc:
            print(f"Invalid selection: {exc}")

        print("Please select a valid model number.")


# ---------------------------------------------------------------------
# Test state
# ---------------------------------------------------------------------

def build_states() -> dict[str, dict[str, Any]]:
    # Conservative schema matching the production forecast state used
    # throughout the V2 tests/smoke flow.
    return {
        "BASELINE": {
            "quality": 87.0,
            "competency": 93.0,
            "attendance": 90.0,
            "release": 60.0,
            "transfer": 9.0,
            "operations_health": 80.0,
        },

        "HIGH_LOAD": {
            "quality": 78.0,
            "competency": 84.0,
            "attendance": 82.0,
            "release": 54.0,
            "transfer": 15.0,
            "operations_health": 68.0,
        },

        "SEVERE_STRESS": {
            "quality": 68.0,
            "competency": 72.0,
            "attendance": 74.0,
            "release": 50.0,
            "transfer": 19.0,
            "operations_health": 55.0,
        },

        "STRONG_STATE": {
            "quality": 94.0,
            "competency": 97.0,
            "attendance": 95.0,
            "release": 72.0,
            "transfer": 5.0,
            "operations_health": 93.0,
        },
    }


# ---------------------------------------------------------------------
# Direct prediction
# ---------------------------------------------------------------------

def run_direct_prediction(
    family: str,
    state: dict[str, Any],
    components: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:

    errors: list[str] = []

    Provider = components["PredictorProvider"]
    PredictionService = components["PredictionService"]

    try:
        provider = Provider()
    except Exception:
        provider = Provider

    # Use whichever model-selection API exists.
    if hasattr(provider, "set_model_family"):
        provider.set_model_family(family)

    if hasattr(provider, "load_pair"):
        provider.load_pair(family)

    try:
        service = PredictionService(provider=provider)
    except TypeError:
        service = PredictionService(provider)

    request_cls = components["PredictionRequest"]

    try:
        request = request_cls(state=state, metadata={})
    except Exception:
        request = request_cls(
            state=state,
            metadata={"model_family": family},
        )

    try:
        result = service.predict(request)
        payload = json_safe(result)

        scan = finite_scan(payload)
        errors.extend(scan)

        return payload, errors

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        return {}, errors


# ---------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------

def run_forecast(
    family: str,
    scenario_name: str,
    horizon: int,
    state: dict[str, Any],
    components: dict[str, Any],
    target_oh: float | None = None,
    target_nps: float | None = None,
) -> tuple[Any, list[str]]:

    errors: list[str] = []

    ForecastOrchestrator = components["ForecastOrchestrator"]
    ForecastRequest = components["ForecastRequest"]

    try:
        orchestrator = ForecastOrchestrator(model_family=family)
    except TypeError:
        try:
            orchestrator = ForecastOrchestrator()
            if hasattr(orchestrator, "set_model_family"):
                orchestrator.set_model_family(family)
        except Exception:
            orchestrator = ForecastOrchestrator()

    params: dict[str, Any] = {
        "state": dict(state),
        "model_family": family,
    }

    if target_oh is not None:
        params["target_oh"] = target_oh

    if target_nps is not None:
        params["target_nps"] = target_nps

    try:
        request = ForecastRequest(
            operation="forecast",
            horizon=horizon,
            scenario=scenario_name,
            parameters=params,
        )
    except Exception:
        request = ForecastRequest(
            operation="forecast",
            horizon=horizon,
            scenario=scenario_name,
            parameters=params,
            model_family=family,
        )

    try:
        result = orchestrator.execute(request)
        payload = json_safe(result)

        errors.extend(finite_scan(payload))

        return payload, errors

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        return {}, errors


# ---------------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------------

def summarize_forecast(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    summary["success"] = get_value(payload, "success")

    errors = get_value(payload, "errors", default=[])
    summary["errors"] = errors

    timeline = (
        get_value(payload, "timeline")
        or get_value(payload, "forecast")
        or get_value(payload, "days")
        or []
    )

    rows = []

    if isinstance(timeline, list):
        for i, day in enumerate(timeline, 1):
            rows.append(
                {
                    "day": i,
                    "operations_health": get_value(
                        day, "operations_health", "operational_health"
                    ),
                    "nps": get_value(day, "nps"),
                    "confidence": get_value(day, "confidence"),
                    "risk": get_value(day, "risk"),
                    "_predicted": get_value(day, "_predicted"),
                }
            )

    summary["timeline"] = rows

    decision = (
        get_value(payload, "decision_intelligence")
        or get_value(payload, "decision_intelligence_v3")
    )

    summary["decision_intelligence"] = decision

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    start = datetime.now(timezone.utc)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    result_path = RESULT_DIR / f"stress_test_result_{now_stamp()}.txt"

    output: list[str] = []

    output.append("AXIPULSEAI V2 — FULL PIPELINE STRESS TEST")
    output.append(f"Started UTC: {start.isoformat()}")
    output.append(f"Project: {PROJECT}")

    try:
        components = load_components()
    except Exception as exc:
        output.append(section("IMPORT ERROR"))
        output.append(f"{type(exc).__name__}: {exc}")
        output.append(traceback.format_exc())
        result_path.write_text("\n".join(output), encoding="utf-8")
        print(f"\nRESULT FILE: {result_path}")
        return 1

    output.append(section("SELECTED MODEL"))
    output.append(f"Model family: {family}")

    states = build_states()

    # Direct prediction tests
    output.append(section("DIRECT OH/NPS PREDICTION TESTS"))

    for name, state in states.items():
        output.append(f"\nINPUT: {name}")
        output.append(compact(state, 3000))

        prediction, errors = run_direct_prediction(
            family,
            state,
            components,
        )

        output.append("\nOUTPUT:")
        output.append(compact(prediction, 4000))

        output.append("\nRESULT:")
        output.append("PASS" if not errors else "FAIL")

        if errors:
            output.append("\nERRORS:")
            output.extend(errors)

    # Forecast scenarios
    scenario_tests = [
        ("BASELINE", "baseline", 3),
        ("BASELINE", "baseline", 5),
        ("BASELINE", "baseline", 7),
        ("HIGH_LOAD", "staffing_shortage", 5),
        ("SEVERE_STRESS", "staffing_shortage", 7),
        ("STRONG_STATE", "baseline", 5),
    ]

    for state_name, scenario_name, horizon in scenario_tests:
        state = states[state_name]

        output.append(
            section(
                f"FORECAST TEST — {state_name} / "
                f"{scenario_name} / H{horizon}"
            )
        )

        output.append("INPUT:")
        output.append(
            compact(
                {
                    "model_family": family,
                    "scenario": scenario_name,
                    "horizon": horizon,
                    "state": state,
                },
                5000,
            )
        )

        payload, errors = run_forecast(
            family=family,
            scenario_name=scenario_name,
            horizon=horizon,
            state=state,
            components=components,
        )

        summary = summarize_forecast(payload)

        output.append("\nOUTPUT:")
        output.append(compact(summary, 12000))

        output.append("\nRESULT:")
        output.append("PASS" if not errors else "FAIL")

        if errors:
            output.append("\nERRORS:")
            output.extend(errors)

    # Target pipeline tests
    target_tests = [
        ("STRONG_STATE", 90.0, 80.0),
        ("BASELINE", 85.0, 80.0),
        ("SEVERE_STRESS", 95.0, 85.0),
    ]

    for state_name, target_oh, target_nps in target_tests:
        state = states[state_name]

        output.append(
            section(
                f"TARGET/DECISION TEST — {state_name} "
                f"target_oh={target_oh} target_nps={target_nps}"
            )
        )

        output.append("INPUT:")
        output.append(
            compact(
                {
                    "model_family": family,
                    "scenario": "baseline",
                    "horizon": 5,
                    "target_oh": target_oh,
                    "target_nps": target_nps,
                    "state": state,
                },
                5000,
            )
        )

        payload, errors = run_forecast(
            family=family,
            scenario_name="baseline",
            horizon=5,
            state=state,
            components=components,
            target_oh=target_oh,
            target_nps=target_nps,
        )

        summary = summarize_forecast(payload)

        output.append("\nOUTPUT:")
        output.append(compact(summary, 12000))

        output.append("\nRESULT:")
        output.append("PASS" if not errors else "FAIL")

        if errors:
            output.append("\nERRORS:")
            output.extend(errors)

    end = datetime.now(timezone.utc)

    output.append(section("FINAL TEST SUMMARY"))
    output.append(f"Started: {start.isoformat()}")
    output.append(f"Finished: {end.isoformat()}")
    output.append(f"Model family: {family}")
    output.append("Pipeline tested:")
    output.append(
        "training-artifact selection → PredictorProvider → "
        "PredictionService → ForecastOrchestrator → "
        "Confidence/Risk → Sensitivity → ADIE V3 → "
        "Recommendations/Strategies"
    )
    output.append(
        "No production source files were modified by this test script."
    )
    output.append(
        f"RESULT FILE: {result_path}"
    )

    result_path.write_text(
        "\n".join(output),
        encoding="utf-8",
    )

    print(f"\nFULL STRESS TEST COMPLETE")
    print(f"MODEL: {family}")
    print(f"RESULT FILE: {result_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())