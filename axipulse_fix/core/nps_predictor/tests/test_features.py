"""
Behavioral tests for NPS feature engineering.

These exercise the real ``prepare_features`` / ``align_features`` paths using a
production-shaped serving row (as produced by ``PredictionService._build_nps_row``):
exact feature names/order, dtype validation, missing/extra features, numeric
coercion, and finite-value guarantees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.forecast_ai.prediction.service import PredictionService
from core.nps_predictor.feature_engineering import align_features, prepare_features


def _serving_row(**overrides):
    svc = PredictionService()
    row = svc._build_nps_row({
        "quality": 85.0, "competency": 90.0, "attendance": 89.0,
        "release": 62.0, "transfer": 8.0, "operations_health": 90.0,
        "total_calls_received": 2000,
    })
    row.update(overrides)
    return row


def _feature_schema():
    return [
        "operational_health", "business_intelligence_factor",
        "member_intelligence_factor", "target_quality", "quality",
        "quality_gap", "target_competency", "competency", "competency_gap",
        "target_attendance", "attendance", "attendance_gap",
        "target_transfer", "transfer", "transfer_gap",
        "total_calls_received", "total_release_calls", "total_surveys",
        "survey_rate", "quality_previous_day", "competency_previous_day",
        "release_previous_day", "transfer_previous_day",
        "attendance_previous_day", "nps_previous_day", "survey_confidence",
    ]


def test_prepare_features_produces_finite_features():
    """prepare_features must produce a deterministic, finite feature set."""
    feats = prepare_features(_serving_row_df())
    numeric = feats.select_dtypes(include=[np.number])
    assert numeric.notna().all().all()
    arr = numeric.to_numpy(dtype=np.float64)
    assert np.isfinite(arr).all()


def _serving_row_df():
    import pandas as pd
    return pd.DataFrame([_serving_row()])


def test_align_features_exact_order():
    """align_features must produce exactly the requested feature order."""
    schema = _feature_schema()
    X = align_features(_serving_row(), schema, {}, None)
    assert list(X.columns) == schema
    assert X.shape[1] == len(schema)


def test_align_features_missing_feature_fills():
    """Missing survey features must use the documented default-fill path (no
    KeyError / no NaN) and still produce the exact feature set."""
    schema = _feature_schema()
    row = _serving_row()
    # total_surveys / survey_rate have explicit default branches in
    # prepare_features; removing them must not break alignment.
    row.pop("total_surveys", None)
    row.pop("survey_rate", None)
    X = align_features(row, schema, {}, None)
    assert list(X.columns) == schema
    assert X.notna().all().all()


def test_align_features_extra_feature_ignored():
    """Unknown extra input features must not leak into the aligned frame."""
    schema = _feature_schema()
    row = _serving_row()
    row["not_a_feature"] = 999.0
    X = align_features(row, schema, {}, None)
    assert list(X.columns) == schema
    assert "not_a_feature" not in X.columns


def test_align_features_numeric_coercion_and_finite():
    """All aligned features must be numeric and finite."""
    schema = _feature_schema()
    row = _serving_row()
    row["quality"] = "85"      # string numeric must be coerced
    row["operational_health"] = "90"
    X = align_features(row, schema, {}, None)
    for col in schema:
        assert pd.api.types.is_numeric_dtype(X[col])
    assert np.isfinite(X.to_numpy()).all()


def test_prepare_features_rejects_unproven_external_factor():
    """An external factor without cutoff-time provenance must be rejected."""
    df = _serving_row_df()
    df["flu_factor"] = [1.0]  # no flu_factor_known_at column
    with pytest.raises(ValueError):
        prepare_features(df)


def test_prepare_features_accepts_cutoff_known_enrollment():
    """enrollment_factor is a documented cutoff-known simulator feature."""
    df = _serving_row_df()
    df["enrollment_factor"] = [1.05]
    feats = prepare_features(df)
    assert "enrollment_factor" in feats.columns
