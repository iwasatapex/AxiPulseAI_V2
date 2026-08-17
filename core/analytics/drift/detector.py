from __future__ import annotations

from dataclasses import dataclass

from .metrics import DriftMetrics


@dataclass(frozen=True)
class DriftDetectionResult:
    drift_detected: bool
    mean_shift_detected: bool
    std_shift_detected: bool
    mean_shift: float
    std_shift: float
    mean_threshold: float
    std_threshold: float
    metadata: dict[str, object]


class DriftDetector:
    """
    Observational drift detector.

    This detector reports statistical change only.
    It does not mutate data, retrain models, or alter predictions.
    """

    @staticmethod
    def detect(
        metrics: DriftMetrics,
        *,
        mean_threshold: float = 0.0,
        std_threshold: float = 0.0,
    ) -> DriftDetectionResult:
        if mean_threshold < 0.0:
            raise ValueError("mean_threshold must be non-negative")

        if std_threshold < 0.0:
            raise ValueError("std_threshold must be non-negative")

        mean_shift_detected = (
            metrics.mean_absolute_shift > mean_threshold
        )

        std_shift_detected = (
            abs(metrics.std_shift) > std_threshold
        )

        return DriftDetectionResult(
            drift_detected=(
                mean_shift_detected or std_shift_detected
            ),
            mean_shift_detected=mean_shift_detected,
            std_shift_detected=std_shift_detected,
            mean_shift=metrics.mean_shift,
            std_shift=metrics.std_shift,
            mean_threshold=float(mean_threshold),
            std_threshold=float(std_threshold),
            metadata={
                "detection_mode": "threshold_based",
                "observational": True,
            },
        )
