"""
Input validation for training rows and prediction rows.

Moved verbatim out of OperationalHealthPredictor in operation_health_predictor.py
(Phase 2, Step 2 — no logic changed).
"""

import numpy as np
import pandas as pd

from .constants import ISSUE_PREFIX


class ValidationMixin:
    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------
    def validate_training(self, row):
        valid, warns = self._validate_common(row)
        if not valid:
            return False, warns
        operations_health = row.get("operational_health")
        if operations_health is None or pd.isna(operations_health):
            return False, ["operational_health is required for training"]
        try:
            operations_health = float(operations_health)
            if not np.isfinite(operations_health):
                return False, ["operational_health must be finite"]
        except Exception:
            return False, ["operational_health must be numeric"]
        return True, warns

    def validate_prediction(self, row):
        valid, warns = self._validate_common(row)
        return valid, warns

    def _validate_common(self, row):
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        warnings_list = []
        percentage_fields = [
            "target_quality", "target_competency", "target_attendance",
            "target_release_rate", "target_transfer_rate",
            "actual_quality", "actual_competency", "actual_attendance",
            "actual_release_rate", "actual_transfer_rate",
        ]
        for field in percentage_fields:
            if field not in row:
                return False, [f"Missing field: {field}"]
            try:
                val = pd.to_numeric(row[field], errors="coerce")
                if pd.isna(val):
                    val = 0.0
                    warnings_list.append(f"{field} was NaN, treated as 0")
                if val < 0 or val > 100:
                    return False, [f"{field} must be between 0 and 100 (got {val})"]
            except Exception:
                return False, [f"{field} must be numeric"]

        calls = row.get("total_calls_received")
        try:
            calls = pd.to_numeric(calls, errors="coerce")
            if pd.isna(calls) or calls <= 0:
                return False, ["total_calls_received must be > 0"]
            if calls < 2000 or calls > 5000:
                warnings_list.append("total_calls_received outside typical range (2000–5000).")
        except Exception:
            return False, ["total_calls_received must be numeric"]

        operational_intelligence_factor = row.get("operational_intelligence_factor", 0)
        try:
            operational_intelligence_factor = pd.to_numeric(operational_intelligence_factor, errors="coerce")
            if pd.isna(operational_intelligence_factor) or operational_intelligence_factor < -100 or operational_intelligence_factor > 100:
                return False, ["operational_intelligence_factor must be between -100 and 100"]
        except Exception:
            return False, ["operational_intelligence_factor must be numeric"]

        issue_cols = [c for c in row.keys() if c.startswith(ISSUE_PREFIX)]
        if issue_cols:
            total = 0
            for c in issue_cols:
                try:
                    val = pd.to_numeric(row[c], errors="coerce")
                    if pd.isna(val):
                        continue
                    total += val
                except Exception:
                    return False, [f"{c} must be numeric"]
            if not np.isclose(total, 100, atol=1.0):
                warnings_list.append(f"Issue types sum to {total:.2f} (not 100). They will be treated as counts.")

        return True, warnings_list


def validate_training(predictor, row):
    """Compatibility API for ValidationMixin.validate_training()."""
    if predictor is None:
        raise TypeError(
            "validate_training() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return ValidationMixin.validate_training(predictor, row)


def validate_prediction(predictor, row):
    """Compatibility API for ValidationMixin.validate_prediction()."""
    if predictor is None:
        raise TypeError(
            "validate_prediction() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return ValidationMixin.validate_prediction(predictor, row)
