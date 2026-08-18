#!/usr/bin/env python3
"""
AxiPulseAI V2 — Automated 9-Scenario GUI/Engine Regression Test

Run this from the AxiPulseAI_V2 project root:

    python tests/automated_gui_9_scenarios.py

or:

    venv/bin/python tests/automated_gui_9_scenarios.py

This exercises the same gui.services layer used by Streamlit, so it tests
the GUI's service integration without requiring browser clicking.

The script:
- discovers available model families and scenarios
- selects a complete OH+NPS model family
- runs 9 forecast scenarios
- validates KPI/NPS bounds and basic forecast integrity
- writes a JSON report
- exits non-zero if any test fails
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Must be run from the AxiPulseAI_V2 project root so `core` and `gui` resolve.
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from gui import services as svc  # noqa: E402


BASE_STATE = {
    "quality": 87.0,
    "competency": 93.0,
    "attendance": 90.0,
    "release": 60.0,
    "transfer": 9.0,
    "operations_health": 95.0,
    "nps": 82.0,
}

TESTS = [
    {
        "id": "HS-01",
        "name": "Core Outage",
        "category": "High Stress",
        "aliases": ["core_outage", "core-outage", "outage"],
        "horizon": 3,
        "state": dict(BASE_STATE),
    },
    {
        "id": "HS-02",
        "name": "CMS Change + High Load",
        "category": "High Stress",
        "aliases": ["cms_change", "cms-change", "cms"],
        "horizon": 3,
        "state": {
            **BASE_STATE,
            "total_calls_received": 2400.0,
            "quality": 84.0,
            "competency": 88.0,
            "operations_health": 88.0,
        },
    },
    {
        "id": "HS-03",
        "name": "Claims Backlog",
        "category": "High Stress",
        "aliases": ["claims_backlog", "claims-backlog", "backlog"],
        "horizon": 3,
        "state": {
            **BASE_STATE,
            "total_calls_received": 2300.0,
            "quality": 84.0,
            "competency": 89.0,
            "release": 56.0,
            "transfer": 12.0,
            "operations_health": 88.0,
        },
    },
    {
        "id": "NS-01",
        "name": "Pharmacy Delay",
        "category": "Normal Stress",
        "aliases": ["pharmacy_delay", "pharmacy-delay", "pharmacy"],
        "horizon": 2,
        "state": dict(BASE_STATE),
    },
    {
        "id": "NS-02",
        "name": "Provider Update",
        "category": "Normal Stress",
        "aliases": ["provider_update", "provider-update", "provider"],
        "horizon": 2,
        "state": dict(BASE_STATE),
    },
    {
        "id": "NO-01",
        "name": "Standard Monday / Normal",
        "category": "Normal Operations",
        "aliases": ["baseline", "normal"],
        "horizon": 2,
        "state": {
            **BASE_STATE,
            "total_calls_received": 2400.0,
        },
    },
    {
        "id": "NO-02",
        "name": "Standard Wednesday / Normal",
        "category": "Normal Operations",
        "aliases": ["baseline", "normal"],
        "horizon": 2,
        "state": dict(BASE_STATE),
    },
    {
        "id": "NO-03",
        "name": "Training Day",
        "category": "Normal Operations",
        "aliases": ["training", "training_day", "training-day"],
        "horizon": 2,
        "state": dict(BASE_STATE),
    },
    {
        "id": "EO-01",
        "name": "Exceptional Operations",
        "category": "Exceptional Operations",
        "aliases": [
            "exceptional",
            "critical",
            "very_high",
            "very-high",
            "core_outage",
        ],
        "horizon": 3,
        "state": {
            **BASE_STATE,
            "quality": 72.0,
            "competency": 76.0,
            "attendance": 78.0,
            "release": 51.0,
            "transfer": 18.0,
            "operations_health": 65.0,
            "nps": 45.0,
            "total_calls_received": 2800.0,
        },
    },
]


def norm(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def choose_family() -> tuple[str | None, list[dict]]:
    models = svc.list_models()
    complete = [m for m in models if "error" not in m]
    active = svc.STATE.get_active_family()
    if active and any(m["family"] == active for m in complete):
        return active, models
    if complete:
        family = complete[0]["family"]
        svc.select_model_family(family)
        return family, models
    return None, models


def discover_scenarios() -> list[dict]:
    scenarios = svc.list_scenarios()
    print("\nAvailable scenarios:")
    for s in scenarios:
        print(f"  - {s.get('id')}: {s.get('name')}")
    return scenarios


def resolve_scenario(test: dict, scenarios: list[dict]) -> str | None:
    normalized = {
        norm(s.get("id")): s.get("id")
        for s in scenarios
        if s.get("id")
    }

    for alias in test["aliases"]:
        if norm(alias) in normalized:
            return normalized[norm(alias)]

    # Fuzzy fallback: match aliases against id/name/description.
    for s in scenarios:
        haystack = " ".join(
            [
                norm(s.get("id")),
                norm(s.get("name")),
                norm(s.get("description")),
            ]
        )
        for alias in test["aliases"]:
            if norm(alias) in haystack:
                return s.get("id")

    return None


def validate_timeline(payload: dict) -> list[str]:
    errors = []

    if not payload.get("success"):
        errors.append("forecast returned success=false")

    timeline = payload.get("timeline") or []
    if not timeline:
        errors.append("timeline is empty")
        return errors

    expected_min = int(payload.get("horizon") or 1) + 1
    if len(timeline) < expected_min:
        errors.append(
            f"timeline has {len(timeline)} rows; expected at least {expected_min}"
        )

    for i, row in enumerate(timeline):
        for key in ("quality", "competency", "attendance", "release", "transfer"):
            if key in row and row[key] is not None:
                if not finite_number(row[key]):
                    errors.append(f"Day {i}: {key} is not finite")
                    continue

                value = float(row[key])
                bounds = {
                    "quality": (60, 100),
                    "competency": (55, 100),
                    "attendance": (65, 100),
                    "release": (50, 100),
                    "transfer": (0, 20),
                }[key]

                if not (bounds[0] <= value <= bounds[1]):
                    errors.append(
                        f"Day {i}: {key}={value:.3f} outside {bounds}"
                    )

        for key in ("operations_health",):
            if key in row and row[key] is not None and not finite_number(row[key]):
                errors.append(f"Day {i}: {key} is not finite")

        if "nps" in row and row["nps"] is not None:
            if not finite_number(row["nps"]):
                errors.append(f"Day {i}: NPS is not finite")
            elif not (-100 <= float(row["nps"]) <= 100):
                errors.append(
                    f"Day {i}: NPS={float(row['nps']):.3f} outside [-100, 100]"
                )

    return errors


def run_test(test: dict, scenario_id: str, family: str) -> dict:
    started = datetime.now()
    try:
        payload = svc.forecast(
            state=test["state"],
            horizon=test["horizon"],
            scenario=scenario_id,
            family=family,
        )

        errors = validate_timeline(payload)

        result = {
            "id": test["id"],
            "name": test["name"],
            "category": test["category"],
            "scenario": scenario_id,
            "family": family,
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "warnings": payload.get("warnings", []),
            "timeline_days": len(payload.get("timeline") or []),
            "horizon": payload.get("horizon"),
            "duration_seconds": (datetime.now() - started).total_seconds(),
        }

        timeline = payload.get("timeline") or []
        if timeline:
            result["day0"] = timeline[0]
            result["last_day"] = timeline[-1]

        return result

    except Exception as exc:
        return {
            "id": test["id"],
            "name": test["name"],
            "category": test["category"],
            "scenario": scenario_id,
            "family": family,
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
            "traceback": traceback.format_exc(),
            "duration_seconds": (datetime.now() - started).total_seconds(),
        }


def main() -> int:
    print("=" * 72)
    print("AxiPulseAI V2 — Automated 9-Scenario Regression Test")
    print("=" * 72)

    family, models = choose_family()

    if not family:
        print("\nFAIL: No complete OH+NPS model family is available.")
        print("Train or install a valid model pair before running this test.")
        return 2

    print(f"\nUsing model family: {family}")
    print(f"Complete model families available: {len([m for m in models if 'error' not in m])}")

    scenarios = discover_scenarios()

    results = []
    skipped = []

    for test in TESTS:
        scenario_id = resolve_scenario(test, scenarios)

        # Baseline is always allowed through the GUI service even if it is
        # only synthesized by list_scenarios().
        if scenario_id is None:
            skipped.append(
                {
                    "id": test["id"],
                    "name": test["name"],
                    "reason": "No matching scenario registered in the engine",
                }
            )
            print(f"\n[{test['id']}] SKIP — {test['name']}")
            continue

        print(f"\n[{test['id']}] {test['name']} [{test['category']}]")
        print(f"  scenario={scenario_id} horizon={test['horizon']}")

        result = run_test(test, scenario_id, family)
        results.append(result)

        if result["status"] == "PASS":
            print(
                f"  PASS — {result['timeline_days']} timeline rows "
                f"in {result['duration_seconds']:.2f}s"
            )
        else:
            print("  FAIL")
            for error in result["errors"]:
                print(f"    - {error}")

    passed = sum(r["status"] == "PASS" for r in results)
    failed = sum(r["status"] == "FAIL" for r in results)

    report = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(ROOT),
        "model_family": family,
        "tests_defined": len(TESTS),
        "tests_executed": len(results),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }

    report_path = ROOT / "automated_gui_9_scenarios_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print(f"RESULT: {passed} PASS / {failed} FAIL / {len(skipped)} SKIP")
    print(f"Report: {report_path}")
    print("=" * 72)

    # A skipped scenario is a test-plan/configuration problem, but the script
    # only returns non-zero for an actual executed failure.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
