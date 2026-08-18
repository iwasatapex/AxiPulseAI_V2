"""
AxiPulseAI – Custom Metrics
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Bucket layout matches core/nps_predictor/inference.py exactly:
#   0..6  = detractor
#   7..8  = passive
#   9..10 = promoter
_DETRACTOR_SLICE = slice(0, 7)
_PASSIVE_SLICE = slice(7, 9)
_PROMOTER_SLICE = slice(9, 11)


def bucket_nps(y) -> np.ndarray:
    """Derive NPS from an (n_rows, 11) array of score-bucket counts.

    Same function is used for both actual and predicted rows, so any
    validation comparison is guaranteed to use an identical definition.
    Rows with total == 0 return NPS 0.0 rather than dividing by zero.
    """
    y = np.asarray(y, dtype=np.float64)
    detractors = y[:, _DETRACTOR_SLICE].sum(axis=1)
    passives = y[:, _PASSIVE_SLICE].sum(axis=1)
    promoters = y[:, _PROMOTER_SLICE].sum(axis=1)
    total = detractors + passives + promoters

    nps = np.zeros_like(total)
    nonzero = total > 0
    nps[nonzero] = (promoters[nonzero] - detractors[nonzero]) / total[nonzero] * 100
    return nps


def compute_nps_error(y_true, y_pred) -> float:
    """MAE between NPS derived from true buckets and NPS derived from
    predicted buckets, using the same 0-6/7-8/9-10 definition production
    inference uses. This is the primary NPS-level validation metric."""
    nps_true = bucket_nps(y_true)
    nps_pred = bucket_nps(y_pred)
    return mean_absolute_error(nps_true, nps_pred)


def calculate_validation_score(y_true, y_pred) -> float:
    """Secondary/diagnostic composite score (bucket-count MAE, bucket-count
    R^2, and NPS MAE). NOT the primary model-selection metric — trainer.py
    selects on compute_nps_error() alone. Kept for diagnostics/back-compat."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nps_mae = compute_nps_error(y_true, y_pred)
    return 0.6 * mae + 0.2 * nps_mae - 0.2 * r2
