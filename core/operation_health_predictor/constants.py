"""
Constants for the AxiPulseAI engine.

Moved verbatim out of operation_health_predictor.py (Phase 2, Step 1). No values changed.
"""

RELEASE_TARGET = 60
TRANSFER_TARGET = 9
ISSUE_PREFIX = "issue_type_"
MODEL_VERSION = "10.10"

DEFAULT_BOUNDS = {
    "actual_quality": (0, 100),
    "actual_competency": (0, 100),
    "actual_attendance": (0, 100),
    "actual_release_rate": (0, 100),
    "actual_transfer_rate": (0, 100),
    "total_calls_received": (1, 20000),
    "operational_intelligence_factor": (-100, 100),
}
ALLOWED_OPTIMIZE_FACTORS = set(DEFAULT_BOUNDS.keys())

REQUIRED_COLUMNS = {
    "date",
    "target_quality", "actual_quality",
    "target_competency", "actual_competency",
    "target_attendance", "actual_attendance",
    "target_release_rate", "actual_release_rate",
    "target_transfer_rate", "actual_transfer_rate",
    "total_calls_received",
    "operational_intelligence_factor",
    "operational_health",
}
