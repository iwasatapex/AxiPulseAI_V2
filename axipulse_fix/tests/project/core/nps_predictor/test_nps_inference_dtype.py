"""
Regression tests for the NPS inference dtype failure.

The trained NPS models (XGBoost / CatBoost / ExtraTrees) require every model
feature to be numeric. A prediction row arriving with ``operational_health``
(or another numeric input) as an ``object`` / ``str`` dtype was previously fed
straight to the model, which rejected it with
"DataFrame.dtypes for data must be int, float, bool or category."

Covers:
- object-typed numeric input is coerced to numeric and prediction succeeds.
- a genuinely non-convertible feature is rejected with a clear error.
- feature ordering exactly matches ``feature_names``.
- the selected model is actually invoked and its prediction is returned.
- no silent fallback when the model fails (predict_single surfaces errors).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

from core.nps_predictor.feature_engineering import align_features
from core.nps_predictor.inference import predict_single

_BASE = {
    "date": "2026-01-15",
    "business_intelligence_factor": 50,
    "member_intelligence_factor": 50,
    "total_calls_received": 1000,
    "actual_release_rate": 60,
    "target_release_rate": 70,
    "total_surveys": 100,
    "survey_rate": 0.1,
    "target_quality": 80,
    "quality": 75,
    "target_competency": 80,
    "competency": 75,
    "target_attendance": 90,
    "attendance": 85,
    "target_transfer": 70,
    "transfer": 65,
}

_FEATURES = [
    "operational_health", "business_intelligence_factor",
    "member_intelligence_factor", "quality_gap", "competency_gap",
    "attendance_gap", "transfer_gap", "total_surveys", "survey_rate",
    "survey_confidence", "is_first_week_of_month", "is_last_week_of_month",
    "days_since_month_start", "days_until_month_end", "month_progress",
    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos", "quarter",
    "is_weekend", "week_of_year", "day_of_month", "is_month_end",
    "is_month_start",
]


def _stats():
    return {c + "_median": 0.0 for c in _FEATURES}


def _fitted_model():
    rng = np.random.default_rng(0)
    model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=5, random_state=1))
    model.fit(
        rng.normal(size=(30, len(_FEATURES))),
        np.abs(rng.normal(5.0, 2.0, size=(30, 11))).astype("float32"),
    )
    return model


class _FakePredictor:
    def __init__(self, model):
        self.model = model
        self.model_name = "ExtraTrees"
        self._all_models = {}
        self.ensemble_weights = None


def test_object_typed_operational_health_is_coerced_to_numeric():
    row = dict(_BASE, operational_health="75.5")  # object/str numeric
    X = align_features(row, _FEATURES, _stats())
    assert pd.api.types.is_numeric_dtype(X["operational_health"])
    assert not (X.dtypes == object).any()
    # Column order exactly matches the training schema.
    assert list(X.columns) == _FEATURES


def test_prediction_succeeds_after_normalization():
    model = _fitted_model()
    row = dict(_BASE, operational_health="75.5")
    X = align_features(row, _FEATURES, _stats())
    # Model is actually invoked and returns 11 outputs.
    pred = model.predict(X)
    assert pred.shape == (1, 11)


def test_predict_single_invokes_model_and_returns_engine_dict():
    predictor = _FakePredictor(_fitted_model())
    X = align_features(dict(_BASE, operational_health=75.5), _FEATURES, _stats())
    result = predict_single(predictor, X, dict(_BASE, operational_health=75.5))
    assert isinstance(result, dict)
    assert "nps" in result
    assert "score_counts" in result
    assert -100.0 <= result["nps"] <= 100.0


def test_non_convertible_feature_is_rejected_clearly():
    row = dict(_BASE, operational_health="Good")
    with pytest.raises(ValueError) as ei:
        align_features(row, _FEATURES, _stats())
    assert "operational_health" in str(ei.value)
    assert "non-numeric" in str(ei.value)


def test_predict_single_surfaces_model_failure_no_silent_fallback():
    # A model that returns a non-11-length output must raise, not fall back.
    class _BadModel:
        def predict(self, X):
            return np.zeros((1, 2))

    predictor = _FakePredictor(_BadModel())
    X = align_features(dict(_BASE, operational_health=75.5), _FEATURES, _stats())
    with pytest.raises(RuntimeError):
        predict_single(predictor, X, dict(_BASE, operational_health=75.5))


def test_categorical_non_feature_column_not_blindly_cast():
    # A history buffer may carry genuinely categorical raw columns (e.g.
    # scenario_regime) that are NOT model features. These must be left untouched
    # (never coerced), while numeric features are still coerced correctly.
    hb = pd.DataFrame([dict(_BASE, scenario_regime="KPI_MET", event_type="foo")])
    row = dict(_BASE, operational_health="75.5")
    X = align_features(row, _FEATURES, _stats(), history_buffer=hb)
    # numeric model feature is numeric; categorical column is not a feature.
    assert pd.api.types.is_numeric_dtype(X["operational_health"])
    assert list(X.columns) == _FEATURES


def test_missing_numeric_feature_is_imputed_and_stays_numeric():
    # A None / missing numeric input must be imputed and remain numeric.
    row = dict(_BASE, operational_health=None)
    X = align_features(row, _FEATURES, _stats())
    assert pd.api.types.is_numeric_dtype(X["operational_health"])
    assert np.isfinite(X.to_numpy(dtype=float)).all()
