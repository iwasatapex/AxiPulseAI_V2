"""Phase-1 temporal contract regression tests.

Verifies the forecasting contract:

    feature_time[T] < target_time[T+1]

for both the OH (operational health) and NPS engines, that the NPS engine only
consumes OH known at cutoff T (never actual/forecast OH(T+1)), that lag/rolling
features stay backward-looking, and that external factors have proven cutoff-time
provenance.

Uses real project code (the shared temporal dataset helper, the actual feature
engineering, and the real forecast service) — no mocking of the central temporal
logic.
"""

import math

import pandas as pd
import pytest

from core.common.temporal_dataset import shift_target_next_day
from core.nps_predictor.feature_engineering import (
    EXTERNAL_FACTOR_COLUMNS,
    prepare_features,
    align_features,
)
from core.forecast_ai.prediction.service import PredictionService


# --------------------------------------------------------------------------- #
# Test 1 & 2 — feature(T) aligned with target(T+1)  (OH single-target, NPS multi)
# --------------------------------------------------------------------------- #
def test_1_oh_target_T_equals_actual_OH_at_T_plus_1():
    times = pd.Series(pd.to_datetime(
        ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    ))
    oh = pd.Series([90.0, 91.0, 92.0, 93.0])
    shifted, target_times = shift_target_next_day(oh, times, field_name="operational_health")
    # Row at T is labeled by the OH realized at T+1, not its own value.
    assert shifted.iloc[0] == 91.0   # OH at index 1
    assert shifted.iloc[1] == 92.0   # OH at index 2
    assert target_times.iloc[0] == times.iloc[1]
    # The mapping is strictly forward: feature_time[T] < target_time[T+1].
    assert target_times.iloc[0] > times.iloc[0]


def test_2_nps_target_T_equals_score_distribution_at_T_plus_1():
    times = pd.Series(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]))
    scores = pd.DataFrame(
        {f"score_{i}": [i + 1, i + 2, i + 3] for i in range(11)}
    )
    shifted, target_times = shift_target_next_day(
        scores, times, field_name="NPS score distribution"
    )
    # Multi-output forward shift: row 0 is labeled by row 1's scores.
    assert shifted.iloc[0].tolist() == [i + 2 for i in range(11)]
    assert target_times.iloc[0] == times.iloc[1]


# --------------------------------------------------------------------------- #
# Test 3 — final source day excluded (no T+1 target)
# --------------------------------------------------------------------------- #
def test_3_final_source_day_has_no_target():
    times = pd.Series(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]))
    oh = pd.Series([90.0, 91.0, 92.0])
    shifted, target_times = shift_target_next_day(oh, times, field_name="operational_health")
    assert math.isnan(shifted.iloc[-1])  # last row has no T+1 target
    assert pd.isna(target_times.iloc[-1])
    assert shifted.notna().sum() == 2  # only rows with a known T+1 target survive


def _nps_state(operations_health=None):
    state = {
        "quality": 82.0,
        "competency": 88.0,
        "attendance": 90.0,
        "transfer": 8.0,
        "release": 60.0,
        "history_buffer": [],
    }
    if operations_health is not None:
        state["operations_health"] = operations_health
    return state


# --------------------------------------------------------------------------- #
# Test 4 — future OH(T+1) cannot enter the NPS feature row at T
# --------------------------------------------------------------------------- #
def test_4_future_OH_Tplus1_cannot_enter_nps_feature_row():
    service = PredictionService()
    known_oh_at_t = 90.0
    state = _nps_state(operations_health=known_oh_at_t)
    # oh here is the OH model forecast — conceptually OH(T+1) — which must NOT
    # be injected into the NPS feature row.
    future_oh_t1 = 97.5
    nps_row = service._build_nps_row(state)
    assert nps_row["operational_health"] == known_oh_at_t
    assert nps_row["operational_health"] != future_oh_t1


def test_4b_state_without_known_oh_does_not_inject_forecast():
    service = PredictionService()
    state = _nps_state()  # no known OH at T
    future_oh_t1 = 97.5
    nps_row = service._build_nps_row(state)
    assert nps_row["operational_health"] != future_oh_t1
    assert nps_row["operational_health"] is None
# --------------------------------------------------------------------------- #
# Test 5 & 6 — lag/rolling features are strictly backward-looking
# --------------------------------------------------------------------------- #
def _oh_df_with_lag():
    dates = pd.to_datetime(pd.date_range("2026-01-01", periods=8))
    return pd.DataFrame(
        {
            "date": dates,
            "actual_quality": [50, 60, 70, 80, 90, 85, 75, 65],
            "actual_competency": [50.0] * 8,
            "actual_attendance": [70.0] * 8,
            "actual_release_rate": [60.0] * 8,
            "actual_transfer_rate": [9.0] * 8,
            "target_quality": [80.0] * 8,
            "target_competency": [90.0] * 8,
            "target_attendance": [75.0] * 8,
            "target_release_rate": [60.0] * 8,
            "target_transfer_rate": [9.0] * 8,
            "total_calls_received": [1000] * 8,
            "operational_intelligence_factor": [0.0] * 8,
        }
    )


def test_5_future_kpi_poisoning_does_not_alter_earlier_feature_rows():
    from core.operation_health_predictor.feature_engineering import FeatureEngineeringMixin
    from core.operation_health_predictor.config import Config

    df = _oh_df_with_lag()
    config = Config(use_lag_features=True, use_cyclical_dates=False, clip_outliers=False)
    mixin = FeatureEngineeringMixin()
    mixin.config = config

    baseline = mixin.prepare_features(df)

    # Poison the FUTURE (last) day's quality to a wildly different value.
    poisoned = df.copy()
    poisoned.loc[poisoned.index[-1], "actual_quality"] = 999.0
    after = mixin.prepare_features(poisoned)

    # Earlier feature rows (including their lag/roll features) are unaffected.
    for col in baseline.columns:
        pd.testing.assert_series_equal(
            baseline[col].iloc[:-1].reset_index(drop=True),
            after[col].iloc[:-1].reset_index(drop=True),
            check_names=False,
        )


def test_6_lag_rolling_features_strictly_backward():
    from core.operation_health_predictor.feature_engineering import FeatureEngineeringMixin
    from core.operation_health_predictor.config import Config

    df = _oh_df_with_lag()
    config = Config(use_lag_features=True, use_cyclical_dates=False, clip_outliers=False)
    mixin = FeatureEngineeringMixin()
    mixin.config = config

    features = mixin.prepare_features(df)

    # actual_quality_lag1 at row T equals actual_quality at row T-1 (backward).
    lag1 = features["actual_quality_lag1"]
    assert lag1.iloc[1] == df["actual_quality"].iloc[0]
    # roll3 at row T is the backward mean over [T-3..T-1] (shifted 1), so it does
    # NOT include row T's own value.
    roll3_col = "actual_quality_roll3"
    assert roll3_col in features.columns
    row_t = 4
    expected = df["actual_quality"].iloc[row_t - 3:row_t].mean()
    assert math.isclose(features[roll3_col].iloc[row_t], expected)


def test_6b_nps_forward_target_uses_next_day_oh_time():
    #
    times = pd.Series(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]))
    oh = pd.Series([90.0, 91.0, 92.0, 93.0])
    shifted, target_times = shift_target_next_day(oh, times, field_name="operational_health")
    assert shifted.iloc[2] == 93.0      # row T=2026-01-07 labeled by OH at T+1
    assert target_times.iloc[2] == times.iloc[3]


# --------------------------------------------------------------------------- #
# Tests 7-9 — external feature provenance
# --------------------------------------------------------------------------- #
def _nps_feature_df(extra_cols=None, known_at=None):
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    data = {
        "date": dates,
        "operational_health": [90.0, 91.0, 92.0],
        "business_intelligence_factor": [0.0] * 3,
        "member_intelligence_factor": [0.0] * 3,
        "target_release_rate": [60.0] * 3,
        "actual_release_rate": [60.0] * 3,
        "total_calls_received": [1000] * 3,
    }
    if extra_cols:
        data.update(extra_cols)
    if known_at:
        data.update(known_at)
    return pd.DataFrame(data)


def test_7_external_feature_unknown_provenance_rejected():
    df = _nps_feature_df(extra_cols={"seasonal_factor": [1.0, 1.1, 1.2]})
    with pytest.raises(ValueError):
        prepare_features(df)


def test_8_external_feature_timestamped_after_T_rejected():
    df = _nps_feature_df(
        extra_cols={"flu_factor": [1.0, 1.0, 1.0]},
        known_at={"flu_factor_known_at": ["2026-01-05", "2026-01-06", "2026-12-31"]},
    )
    with pytest.raises(ValueError):
        prepare_features(df)


def test_9_external_feature_known_at_or_before_T_accepted():
    df = _nps_feature_df(
        extra_cols={"holiday_factor": [1.0, 1.0, 1.0]},
        known_at={"holiday_factor_known_at": ["2026-01-05", "2026-01-05", "2026-01-06"]},
    )
    features = prepare_features(df)
    # Holiday factor admitted because it is provably known at/before cutoff.
    assert "holiday_factor" in features.columns


# --------------------------------------------------------------------------- #
# Test 10 — training and inference use the same temporal contract
# --------------------------------------------------------------------------- #
def test_10_inference_align_features_shares_external_provenance_contract():
    # align_features (inference path) calls prepare_features internally, so the
    # same cutoff-provenance guard applies at inference as at training.
    row = {
        "date": "2026-01-05",
        "operational_health": 90.0,
        "business_intelligence_factor": 0.0,
        "member_intelligence_factor": 0.0,
        "target_release_rate": 60.0,
        "actual_release_rate": 60.0,
        "total_calls_received": 1000,
        "seasonal_factor": 1.2,  # present but no proven-known-at timestamp
    }
    with pytest.raises(ValueError):
                align_features(
            row, feature_names=["seasonal_factor"], feature_stats={}, history_buffer=None
        )


# --------------------------------------------------------------------------- #
# Test 11 — serving-time: align_features must not crash on minimal row
# --------------------------------------------------------------------------- #
def test_11_align_features_handles_minimal_serving_row():
    """
    P3 regression: the serving path (``_build_nps_row``) produces a minimal
    row dict whose columns do not include KPI columns such as
    ``quality``, ``competency``, ``attendance``, ``transfer``.

    ``prepare_features`` previously raised ``KeyError`` because the base
    feature list unconditionally referenced columns missing from the
    single-row serving DataFrame.  The fix must guarantee that all base
    feature columns are materialised with a default of 0.0 before the
    column-selection return statement.
    """
    service = PredictionService()
    state = _nps_state(operations_health=90.0)
    row = service._build_nps_row(state)

    # The row should NOT contain any NPS target columns (same-day outcomes).
    for target_col in [
        "nps_today", "promoter_pct", "passive_pct", "detractor_pct",
        "nps",
    ]:
        assert target_col not in row, (
            f"Serving row must not fabricate same-day target '{target_col}'"
        )

    # align_features must succeed on the minimal row and produce a feature
    # DataFrame whose columns are exactly the (non-roll) feature schema.
    feature_names = [
        "operational_health", "quality", "quality_gap",
        "target_quality", "competency", "competency_gap",
        "target_competency", "attendance", "attendance_gap",
        "target_attendance", "transfer", "transfer_gap",
        "target_transfer", "total_surveys", "survey_rate",
        "survey_confidence",
        "is_first_week_of_month", "is_last_week_of_month",
        "days_since_month_start", "days_until_month_end",
        "month_progress",
    ]
    X = align_features(row, feature_names, {}, None)

    # No target columns survived into the aligned feature matrix.
    for target_col in [
        "nps_today", "promoter_pct", "passive_pct", "detractor_pct",
        "score_0", "score_10",
    ]:
        assert target_col not in X.columns, (
            f"Target column '{target_col}' must not appear in aligned features"
        )

    # Every requested feature column is present.
    for fc in feature_names:
        assert fc in X.columns


# --------------------------------------------------------------------------- #
# Test 12 — serving row must NOT carry a realisation of the T-day NPS outcome
# as a model input
# --------------------------------------------------------------------------- #
def test_12_serving_row_has_no_same_day_nps_target():
    """
    The row dict returned by ``_build_nps_row`` is what the NPS predictor
    receives.  It must contain only T-known information and must NOT include
    the same-day NPS realisation (``nps``, ``nps_today``, ``promoter_pct``,
    ``score_*``) as a model input.
    """
    service = PredictionService()
    state = _nps_state(operations_health=90.0)
    # Simulate a state that happens to carry today's realisation.
    state["nps"] = 85.0
    state["nps_today"] = 85.0
    state["promoter_pct"] = 70.0
    state["score_0"] = 3
    state["score_10"] = 7
    row = service._build_nps_row(state)

    # The same-day realisation values from ``state`` must NOT appear in the
    # serving row as model features.
    target_keys = ["nps", "nps_today", "promoter_pct", "passive_pct",
                   "detractor_pct", "score_0", "score_10"]
    for k in target_keys:
        assert k not in row, (
            f"Same-day target '{k}' must not appear in serving row"
        )

    # The OH feature must be the value known at T, never the T+1 forecast.
    assert row["operational_health"] == 90.0
