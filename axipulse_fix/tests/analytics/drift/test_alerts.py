import pytest

from core.analytics.drift import (
    DriftAlert,
    aggregate_drift_alerts,
    alert_from_signal,
)


def test_no_detected_drift_produces_none_severity():
    result = aggregate_drift_alerts(
        [
            DriftAlert(
                name="prediction",
                detected=False,
                severity="none",
                message="No prediction drift.",
            ),
            DriftAlert(
                name="calibration",
                detected=False,
                severity="none",
                message="No calibration drift.",
            ),
        ]
    )

    assert result.drift_detected is False
    assert result.detected_count == 0
    assert result.severity == "none"
    assert len(result.alerts) == 2


def test_warning_drift_is_aggregated():
    result = aggregate_drift_alerts(
        [
            DriftAlert(
                name="prediction",
                detected=True,
                severity="warning",
                message="Prediction distribution shifted.",
            ),
            DriftAlert(
                name="calibration",
                detected=False,
                severity="none",
                message="Calibration stable.",
            ),
        ]
    )

    assert result.drift_detected is True
    assert result.detected_count == 1
    assert result.severity == "warning"


def test_critical_has_priority():
    result = aggregate_drift_alerts(
        [
            DriftAlert(
                name="prediction",
                detected=True,
                severity="warning",
                message="Prediction drift.",
            ),
            DriftAlert(
                name="calibration",
                detected=True,
                severity="critical",
                message="Calibration drift.",
            ),
        ]
    )

    assert result.drift_detected is True
    assert result.detected_count == 2
    assert result.severity == "critical"


def test_info_has_lower_priority_than_warning():
    result = aggregate_drift_alerts(
        [
            DriftAlert(
                name="prediction",
                detected=True,
                severity="info",
                message="Prediction signal.",
            ),
            DriftAlert(
                name="calibration",
                detected=True,
                severity="warning",
                message="Calibration signal.",
            ),
        ]
    )

    assert result.severity == "warning"


def test_alert_from_signal_false_becomes_none():
    alert = alert_from_signal(
        "prediction",
        False,
        severity="critical",
        message="No drift.",
    )

    assert alert.detected is False
    assert alert.severity == "none"


def test_alert_from_signal_true_preserves_severity():
    alert = alert_from_signal(
        "prediction",
        True,
        severity="critical",
        message="Drift detected.",
    )

    assert alert.detected is True
    assert alert.severity == "critical"


def test_invalid_severity_is_rejected():
    with pytest.raises(ValueError):
        aggregate_drift_alerts(
            [
                DriftAlert(
                    name="prediction",
                    detected=True,
                    severity="bad",
                    message="Invalid.",
                )
            ]
        )


def test_empty_name_is_rejected():
    with pytest.raises(ValueError):
        aggregate_drift_alerts(
            [
                DriftAlert(
                    name="",
                    detected=True,
                    severity="warning",
                    message="Invalid.",
                )
            ]
        )


def test_empty_message_is_rejected():
    with pytest.raises(ValueError):
        aggregate_drift_alerts(
            [
                DriftAlert(
                    name="prediction",
                    detected=True,
                    severity="warning",
                    message="",
                )
            ]
        )


def test_aggregation_does_not_mutate_input():
    alerts = [
        DriftAlert(
            name="prediction",
            detected=True,
            severity="warning",
            message="Prediction drift.",
        )
    ]

    before = list(alerts)

    result = aggregate_drift_alerts(alerts)

    assert alerts == before
    assert result.alerts == tuple(before)


def test_metadata_is_observational():
    result = aggregate_drift_alerts([])

    assert result.metadata["aggregation_mode"] == (
        "existing_signal_aggregation"
    )
    assert result.metadata["observational"] is True
