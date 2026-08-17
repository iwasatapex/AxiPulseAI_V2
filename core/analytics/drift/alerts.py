from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DriftAlert:
    name: str
    detected: bool
    severity: str
    message: str


@dataclass(frozen=True)
class DriftAlertSummary:
    alerts: tuple[DriftAlert, ...]
    detected_count: int
    severity: str
    drift_detected: bool
    metadata: dict[str, object]


def _normalise_alert(
    name: str,
    detected: bool,
    severity: str,
    message: str,
) -> DriftAlert:
    clean_name = str(name).strip()
    clean_severity = str(severity).strip().lower()
    clean_message = str(message).strip()

    if not clean_name:
        raise ValueError("alert name must not be empty")

    if clean_severity not in {
        "none",
        "info",
        "warning",
        "critical",
    }:
        raise ValueError(
            "severity must be one of: none, info, warning, critical"
        )

    if not clean_message:
        raise ValueError("alert message must not be empty")

    return DriftAlert(
        name=clean_name,
        detected=bool(detected),
        severity=clean_severity,
        message=clean_message,
    )


def aggregate_drift_alerts(
    alerts: Iterable[DriftAlert],
) -> DriftAlertSummary:
    """
    Aggregate already-computed drift signals.

    Observational only:
    - does not run predictors
    - does not modify models
    - does not modify datasets
    - does not retrain
    - does not replace model artifacts
    """
    normalised = tuple(
        _normalise_alert(
            alert.name,
            alert.detected,
            alert.severity,
            alert.message,
        )
        for alert in alerts
    )

    detected = tuple(
        alert
        for alert in normalised
        if alert.detected
    )

    severity_rank = {
        "none": 0,
        "info": 1,
        "warning": 2,
        "critical": 3,
    }

    if not detected:
        aggregate_severity = "none"
    else:
        aggregate_severity = max(
            (alert.severity for alert in detected),
            key=lambda value: severity_rank[value],
        )

    return DriftAlertSummary(
        alerts=normalised,
        detected_count=len(detected),
        severity=aggregate_severity,
        drift_detected=bool(detected),
        metadata={
            "aggregation_mode": "existing_signal_aggregation",
            "observational": True,
        },
    )


def alert_from_signal(
    name: str,
    detected: bool,
    *,
    severity: str = "warning",
    message: str = "Drift signal evaluated.",
) -> DriftAlert:
    """
    Convert one existing drift result into an aggregatable alert.
    """
    return _normalise_alert(
        name=name,
        detected=detected,
        severity=severity if detected else "none",
        message=message,
    )
