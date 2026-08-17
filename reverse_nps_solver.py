#!/usr/bin/env python3
"""
Reverse NPS solver.

Loads the canonical ``NPSPredictor`` (the same loader the working CLI and
ForecastAI prediction paths use) and searches the KPI grid for combinations
that drive the predicted NPS toward a target. When the current model cannot
reach the target (insufficient sensitivity / unreachable range) it reports a
truthful ``found=False`` with a reason - it never fabricates a solution.
"""

import itertools
import math
import pandas as pd
from pathlib import Path

from core.nps_predictor.predictor import NPSPredictor


MODEL = Path("models/nps_predictor_model.pkl")
DATA = Path("training/training.csv")

# Business targets mirror the canonical ``PredictionService._build_nps_row``.
TARGET_KPIS = {
    "target_quality": 87.0,
    "target_competency": 93.0,
    "target_attendance": 90.0,
    "target_release_rate": 60.0,
    "target_transfer_rate": 9.0,
}

# KPI sweep ranges (kept from the original solver).
QUALITY_RANGE = range(80, 101, 2)
COMPETENCY_RANGE = range(85, 101, 2)
RELEASE_RANGE = range(55, 71, 2)
TRANSFER_RANGE = range(5, 15, 1)
ATTENDANCE_RANGE = range(85, 96, 2)

TARGET_MIN, TARGET_MAX = -100.0, 100.0
# ~one survey step at typical survey volumes (each survey shifts NPS by
# 100 / total_surveys ≈ 0.8 points).
_DEFAULT_TOLERANCE = 1.0

_loaded_predictor = None


def load_latest_state():

    df = pd.read_csv(DATA)

    row = df.tail(1).iloc[0]

    return {
        "operational_health": float(row["operational_health"]),
        "quality": float(row["actual_quality"]),
        "competency": float(row["actual_competency"]),
        "attendance": float(row["actual_attendance"]),
        "release": float(row["actual_release_rate"]),
        "transfer": float(row["actual_transfer_rate"]),
        "total_calls_received": float(row["total_calls_received"]),
    }


def _load_predictor():
    """Load the canonical NPSPredictor used by the CLI / ForecastAI path.

    The bundle is loaded through ``NPSPredictor.load_model`` so the returned
    object is a predictor, never the raw joblib bundle dict.
    """
    global _loaded_predictor
    if _loaded_predictor is None:
        predictor = NPSPredictor()
        predictor.load_model(str(MODEL))
        _loaded_predictor = predictor
    return _loaded_predictor


def predict(predictor, state):

    """Run the canonical engine for a KPI state and return the NPS scalar."""

    result = predictor.predict(state)

    return float(result["nps"])


def _build_state(base, quality, competency, release, transfer, attendance):
    """Build the canonical NPS feature row for a KPI combination."""
    state = {
        "operational_health": base["operational_health"],
        "total_calls_received": base["total_calls_received"],
    }
    state.update(TARGET_KPIS)
    state.update({
        "quality": float(quality),
        "competency": float(competency),
        "attendance": float(attendance),
        "actual_release_rate": float(release),
        "transfer_rate": float(transfer),
        "actual_transfer_rate": float(transfer),
        "total_surveys": max(
            1,
            int(
                base["total_calls_received"]
                * float(release) / 100.0
                * 0.10
            )
        ),
        "survey_rate": 0.10,
    })
    return state



def solve_for(
    target: float,
    tolerance: float = None,
    max_combinations: int = None,
) -> dict:
    """Reverse-optimise the KPIs that drive a target NPS.

    Returns a dict with the best KPI combination found, the predicted NPS,
    the search distance and - when no combination reaches the target - a
    truthful ``found=False`` plus a reason. Reusable from the GUI and the
    CLI (``solve``).
    """
    try:
        target = float(target)
    except (TypeError, ValueError):
        return {
            "found": False,
            "distance": None,
            "target": target,
            "reason": "target must be numeric",
        }

    if not math.isfinite(target):
        return {
            "found": False,
            "distance": None,
            "target": target,
            "reason": "target must be finite",
        }

    if not (TARGET_MIN <= target <= TARGET_MAX):
        return {
            "found": False,
            "distance": None,
            "target": target,
            "reason": f"target must be within [{TARGET_MIN}, {TARGET_MAX}]",
        }

    if tolerance is None:
        tolerance = _DEFAULT_TOLERANCE

    predictor = _load_predictor()

    base = load_latest_state()

    best = None
    best_distance = float("inf")
    evaluated = 0

    print("\nSearching KPI combinations...\n")

    for q, c, r, t, a in itertools.product(
        QUALITY_RANGE,
        COMPETENCY_RANGE,
        RELEASE_RANGE,
        TRANSFER_RANGE,
        ATTENDANCE_RANGE,
    ):
        if max_combinations is not None and evaluated >= max_combinations:
            break
        evaluated += 1

        state = _build_state(base, q, c, r, t, a)

        try:
            nps = predict(predictor, state)
        except Exception:
            continue

        distance = abs(nps - target)

        if distance < best_distance:
            best_distance = distance
            best = {
                "predicted_nps": nps,
                "quality": q,
                "competency": c,
                "release": r,
                "transfer": t,
                "attendance": a,
                "operational_health": base["operational_health"],
            }
            if best_distance <= tolerance:
                break

    if best is None:
        return {
            "found": False,
            "distance": None,
            "target": target,
            "error": "No solution found",
            "reason": "no KPI combination produced a valid NPS prediction",
        }

    if best_distance > tolerance:
        result = dict(best)
        result["found"] = False
        result["distance"] = round(best_distance, 3)
        result["target"] = target
        result["error"] = "No solution found"
        result["reason"] = (
            "target NPS is outside the current model's achievable range / "
            f"sensitivity (closest predicted NPS {best['predicted_nps']:.2f}, "
            f"distance {best_distance:.3f} > tolerance {tolerance:.3f})"
        )
        return result

    result = dict(best)
    result["found"] = True
    result["distance"] = round(best_distance, 3)
    result["target"] = target
    result["tolerance"] = float(tolerance)
    return result


def solve():

    target = float(
        input("Target NPS: ")
    )

    best = solve_for(target)

    print("\n==============================")
    print("REQUIRED ACTUALS")
    print("==============================")

    if best.get("found"):

        for k, v in best.items():
            print(
                f"{k:25}: {v}"
            )

    else:
        print("No solution found")
        if best.get("reason"):
            print(f"Reason: {best['reason']}")


if __name__ == "__main__":
    solve()

