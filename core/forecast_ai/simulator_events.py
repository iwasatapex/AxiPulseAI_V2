"""Canonical V2.3 simulator event specification (data constants).

This module encodes the canonical V2.3 simulation-day event contract: exactly
one mutually-exclusive event is selected per simulation day, with a fixed
probability and fixed absolute-point effects on the KPI state and call volume.

This is a SPECIFICATION / constants module only. It does not implement a live
OH/NPS simulator and does not feed the production Forecast path (which is
driven by the trained CatBoost OH / XGBoost-NPS models). It documents the
V2.3 event model so it can be verified against the canonical specification
without confusing it with ``core.forecast_ai.scenarios`` (the Forecast AI
scenario system).

Effects are ABSOLUTE points (additive deltas), not percentages.
``calls`` is a multiplier (e.g. 1.05 = +5% call volume).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Exactly one mutually-exclusive daily event.
EVENT_NAMES: Tuple[str, ...] = (
    "NORMAL",
    "PHARMACY_DELAY",
    "PROVIDER_UPDATE",
    "CLAIMS_BACKLOG",
    "SYSTEM_SLOWDOWN",
    "CORE_OUTAGE",
    "CMS_CHANGE",
    "TRAINING",
)

# Per-event selection probability (must sum to 1.0).
EVENT_PROBABILITIES: Dict[str, float] = {
    "NORMAL": 0.40,
    "PHARMACY_DELAY": 0.10,
    "PROVIDER_UPDATE": 0.20,
    "CLAIMS_BACKLOG": 0.10,
    "SYSTEM_SLOWDOWN": 0.03,
    "CORE_OUTAGE": 0.02,
    "CMS_CHANGE": 0.05,
    "TRAINING": 0.10,
}

# Effects are ABSOLUTE points on quality(Q)/competency(C)/release(R)/
# transfer(T)/attendance/OH, plus a call-volume multiplier.
# Keys: quality, competency, release, transfer, attendance, operations_health, calls.
_EFFECT_KEYS: Tuple[str, ...] = (
    "quality",
    "competency",
    "release",
    "transfer",
    "attendance",
    "operations_health",
    "calls",
)

EVENT_EFFECTS: Dict[str, Dict[str, float]] = {
    "NORMAL": {"quality": 0.0, "competency": 0.0, "release": 0.0,
               "transfer": 0.0, "attendance": 0.0, "operations_health": 0.0,
               "calls": 1.00},
    "PHARMACY_DELAY": {"quality": -2.0, "competency": -2.0, "release": -2.0,
                       "transfer": 2.0, "attendance": 0.0, "operations_health": -0.5,
                       "calls": 1.05},
    "PROVIDER_UPDATE": {"quality": -1.0, "competency": -1.0, "release": -1.0,
                        "transfer": 1.0, "attendance": 0.0, "operations_health": -0.3,
                        "calls": 1.03},
    "CLAIMS_BACKLOG": {"quality": -4.0, "competency": -3.0, "release": -4.0,
                       "transfer": 4.0, "attendance": 0.0, "operations_health": -0.8,
                       "calls": 1.08},
    "SYSTEM_SLOWDOWN": {"quality": -5.0, "competency": -2.0, "release": -3.0,
                        "transfer": 4.0, "attendance": -2.0, "operations_health": -0.6,
                        "calls": 1.04},
    "CORE_OUTAGE": {"quality": -10.0, "competency": -6.0, "release": -8.0,
                    "transfer": 6.0, "attendance": -5.0, "operations_health": -1.5,
                    "calls": 1.12},
    "CMS_CHANGE": {"quality": -7.0, "competency": -9.0, "release": -6.0,
                   "transfer": 7.0, "attendance": -3.0, "operations_health": -1.2,
                   "calls": 1.15},
    "TRAINING": {"quality": 3.0, "competency": 4.0, "release": 2.0,
                 "transfer": -2.0, "attendance": 0.0, "operations_health": 0.5,
                 "calls": 0.90},
}

# The canonical V2.3 event-precedence order (highest precedence first).
EVENT_PRECEDENCE: Tuple[str, ...] = (
    "hard_business_constraints",
    "event_effects",
    "oh_effects",
    "intelligence_effects",
    "complexity_effects",
    "momentum_recovery",
    "random_variation",
)


def event_names() -> List[str]:
    """Return the ordered list of V2.3 event names."""
    return list(EVENT_NAMES)


def event_effect_keys() -> List[str]:
    """Return the ordered effect keys."""
    return list(_EFFECT_KEYS)


def total_probability() -> float:
    """Sum of all event probabilities (must equal 1.0)."""
    return sum(EVENT_PROBABILITIES.values())
