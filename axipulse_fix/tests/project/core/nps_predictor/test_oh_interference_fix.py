"""Regression tests for OH interference removal in the NPS predictor.

Root cause fixed: ``postprocess_predictions`` previously included a
hard-coded ``health_component = (ohi/80.0) * 10.0`` in the rule-based
``target_nps`` formula, injecting artificial OH -> NPS monotonic causality
on top of the trained model's learned OH feature. The fix makes the rule
health-neutral (health_component = 10.0), removing the artificial forcing
while preserving ``operational_health`` as a legitimate trained feature
(see PredictionService._build_nps_row / _known_oh_at_cutoff).
"""

import numpy as np

from core.nps_predictor.inference import postprocess_predictions


def _flat_ml_distribution():
    return np.full(11, 1.0 / 11.0)


def _row(oh):
    return {
        "total_calls_received": 2000,
        "actual_release_rate": 60.0,
        "target_quality": 87,
        "quality": 85,
        "target_competency": 93,
        "competency": 88,
        "target_attendance": 90,
        "attendance": 90,
        "target_release_rate": 60,
        "target_transfer_rate": 9,
        "transfer_rate": 9,
        "business_intelligence_factor": 0,
        "member_intelligence_factor": 0,
        "operational_health": oh,
    }


def test_oh_does_not_monotonically_force_nps():
    """With a flat ML distribution (no ML signal), NPS must not rise
    monotonically with OH through the rule component."""
    nps_low = postprocess_predictions(_flat_ml_distribution(), _row(70))["nps"]
    nps_high = postprocess_predictions(_flat_ml_distribution(), _row(90))["nps"]
    nps_very_high = postprocess_predictions(
        _flat_ml_distribution(), _row(110)
    )["nps"]

    # The rule is health-neutral: a 40-point OH increase must NOT produce a
    # large monotonic NPS lift (before the fix it added ~+4 NPS for 80->110).
    assert nps_high - nps_low < 5.0
    assert nps_very_high - nps_low < 8.0


def test_oh_still_enters_nps_as_legitimate_feature():
    """operational_health must remain a legitimate NPS feature (the temporal
    contract uses it as known-at-cutoff OH); the fix only removed the
    hard-coded rule override, not the trained feature."""
    from core.forecast_ai.prediction.service import PredictionService

    service = PredictionService()
    state = {
        "quality": 85.0,
        "competency": 88.0,
        "attendance": 90.0,
        "transfer": 9.0,
        "release": 60.0,
        "operations_health": 90.0,
        "history_buffer": [],
    }
    nps_row = service._build_nps_row(state)
    assert nps_row["operational_health"] == 90.0
