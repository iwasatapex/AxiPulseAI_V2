"""Behavioral regression: temporal integrity / provenance / cutoff contract.

These tests exercise the REAL temporal contract end to end:

* explicit cutoff from request/state (historical replay, backtesting,
  future-dated simulation) — NOT fabricated ``date.today()``
* recursive OH -> NPS state marked predicted, never observed
* OH=0 is a valid value (no ``value or fallback`` truthiness bug)
* repeated dates stay causally valid
"""
from __future__ import annotations

import datetime

import pytest


def _state(operations_health=None, **overrides):
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
    state.update(overrides)
    return state


# --------------------------------------------------------------------------- #
# Explicit cutoff (Task A)
# --------------------------------------------------------------------------- #

def test_cutoff_is_explicit_from_state_not_fabricated():
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    explicit = "2024-03-15"
    row = svc._build_oh_row(_state(operations_health=90.0, date=explicit))
    assert row["date"] == explicit

    nps_row = svc._build_nps_row(_state(operations_health=90.0, date=explicit))
    assert nps_row["date"] == explicit


def test_cutoff_accepts_cutoff_key_and_datetime():
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    row = svc._build_oh_row(_state(operations_health=90.0, cutoff="2024-06-01"))
    assert row["date"] == "2024-06-01"

    row2 = svc._build_oh_row(
        _state(operations_health=90.0, date="2024-07-04T12:00:00"))
    assert row2["date"].startswith("2024-07-04")


def test_invalid_cutoff_rejected():
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    with pytest.raises(ValueError, match="[Cc]utoff"):
        svc._build_oh_row(_state(operations_health=90.0, date="not-a-date"))


def test_default_cutoff_is_today_for_normal_production():
    """No explicit cutoff -> today (normal production forecasting)."""
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    row = svc._build_oh_row(_state(operations_health=90.0))
    assert row["date"] == datetime.date.today().isoformat()


def test_forecast_orchestrator_propagates_explicit_cutoff():
    """Forecast start date reflects an explicit request cutoff, enabling
    historical replay / future-dated simulation."""
    from core.forecast_ai.engines.forecast_orchestrator import (
        ForecastOrchestrator,
    )

    orch = ForecastOrchestrator()
    explicit = "2023-11-20"
    assert orch._explicit_cutoff(
        type("R", (), {"parameters": {"cutoff": explicit}})(), {}
    ) == explicit
    assert orch._explicit_cutoff(
        type("R", (), {"parameters": {}})(), {"date": explicit}
    ) == explicit
    assert orch._explicit_cutoff(
        type("R", (), {"parameters": {}})(), {}
    ) is None


# --------------------------------------------------------------------------- #
# Recursive OH -> NPS predicted-vs-observed state (Task E)
# --------------------------------------------------------------------------- #

def test_predicted_recursive_state_never_treated_as_observed():
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    # A predicted recursive step must not yield a known-at-cutoff OH.
    predicted = _state(operations_health=95.0, _predicted=True)
    assert svc._known_oh_at_cutoff(predicted) is None

    # An observed state's OH is trusted as known-at-cutoff.
    observed = _state(operations_health=95.0, _predicted=False)
    assert svc._known_oh_at_cutoff(observed) == 95.0


def test_predicted_history_rows_excluded_from_observed_oh():
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    state = _state(operations_health=80.0, history_buffer=[
        {"operations_health": 90.0, "_predicted": True},
        {"operations_health": 88.0, "_predicted": False},
    ])
    # 88.0 is the latest OBSERVED row; the predicted 90.0 must be ignored.
    assert svc._known_oh_at_cutoff(state) == 88.0


def test_nps_row_uses_known_oh_not_recursive_forecast():
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    state = _state(operations_health=90.0)
    nps_row = svc._build_nps_row(state)
    assert nps_row["operational_health"] == 90.0


# --------------------------------------------------------------------------- #
# OH=0 is a valid value (Task F)
# --------------------------------------------------------------------------- #

def test_oh_zero_is_a_valid_known_value():
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    # OH=0 must be treated as a real observed value, not as "missing".
    state = _state(operations_health=0.0, _predicted=False)
    assert svc._known_oh_at_cutoff(state) == 0.0

    # And it must be carried into the NPS feature row as 0.0, not fallback.
    nps_row = svc._build_nps_row(state)
    assert nps_row["operational_health"] == 0.0


def test_oh_zero_in_observed_history_wins():
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    state = _state(operations_health=None, history_buffer=[
        {"operations_health": 0.0, "_predicted": False},
    ])
    assert svc._known_oh_at_cutoff(state) == 0.0


# --------------------------------------------------------------------------- #
# Repeated dates / trajectory semantics (Task G)
# --------------------------------------------------------------------------- #

def test_repeated_dates_alignment_requires_trajectory_or_occurence():
    from core.common.temporal_dataset import shift_target_next_day
    import pandas as pd

    times = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]))
    target = pd.Series([10.0, 11.0, 20.0, 21.0])
    shifted, target_times = shift_target_next_day(
        target, times, field_name="t"
    )
    # Each occurrence on Jan 1 pairs with the same occurrence on Jan 2.
    assert shifted.iloc[0] == 20.0
    assert shifted.iloc[1] == 21.0
    # Final day has no T+1 target.
    assert pd.isna(shifted.iloc[2]) and pd.isna(shifted.iloc[3])


def test_repeated_dates_with_explicit_trajectory_ids():
    from core.common.temporal_dataset import shift_target_next_day
    import pandas as pd

    times = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]))
    target = pd.Series([10.0, 11.0, 20.0, 21.0])
    trajectory_ids = ["A", "B", "A", "B"]
    shifted, _ = shift_target_next_day(
        target, times, trajectory_ids=trajectory_ids, field_name="t"
    )
    # Trajectory A: 10 -> 20 ; trajectory B: 11 -> 21.
    assert shifted.iloc[0] == 20.0
    assert shifted.iloc[1] == 21.0


def test_non_monotonic_dates_rejected():
    from core.common.temporal_dataset import shift_target_next_day
    import pandas as pd
    import pytest as _p

    times = pd.Series(pd.to_datetime(["2026-01-02", "2026-01-01"]))
    with _p.raises(ValueError, match="[Mm]onotonic"):
        shift_target_next_day(pd.Series([1.0, 2.0]), times, field_name="t")


# --------------------------------------------------------------------------- #
# Missing / invalid provenance (Task B)
# --------------------------------------------------------------------------- #

def test_missing_provenance_is_not_silently_observed():
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    # A predicted recursive state (no real provenance) yields no known OH.
    assert svc._known_oh_at_cutoff(_state(_predicted=True)) is None
    # A state with no operations_health and no observed history yields none.
    assert svc._known_oh_at_cutoff(_state()) is None


def test_known_at_cutoff_rejects_future_feature():
    from core.common.temporal_contract import assert_known_at_cutoff
    import pytest as _p

    cutoff = "2026-01-15T00:00:00+00:00"
    with _p.raises(ValueError, match="[Ll]eakage"):
        assert_known_at_cutoff("2026-01-16T00:00:00+00:00", cutoff, field_name="x")
    # At/before cutoff is fine.
    assert_known_at_cutoff("2026-01-15T00:00:00+00:00", cutoff, field_name="x")
    assert_known_at_cutoff("2026-01-14T00:00:00+00:00", cutoff, field_name="x")


def test_forecast_boundary_requires_strictly_after():
    from core.common.temporal_contract import assert_forecast_boundary
    import pytest as _p

    cutoff = "2026-01-15T00:00:00+00:00"
    with _p.raises(ValueError, match="strictly after"):
        assert_forecast_boundary(cutoff, "2026-01-15T00:00:00+00:00")
    assert_forecast_boundary(cutoff, "2026-01-16T00:00:00+00:00")


# --------------------------------------------------------------------------- #
# Recursive OH -> NPS (observed OH -> NPS H1, predicted OH H1 -> NPS H2)
# --------------------------------------------------------------------------- #

def test_recursive_predicted_oh_feeds_next_horizon_not_as_observed():
    """The intended recursive contract: a predicted OH for a later horizon
    must be passed forward as PREDICTED state (never silently observed).

    Concretely: a state flagged ``_predicted`` (recursive day >= 2) must not
    yield a known-at-cutoff OH to an NPS feature row as if it were observed.
    """
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()

    # H1: observed state -> known OH used as NPS input.
    h1 = _state(operations_health=82.0, _predicted=False)
    assert svc._known_oh_at_cutoff(h1) == 82.0
    assert svc._build_nps_row(h1)["operational_health"] == 82.0

    # H2: recursive predicted state -> must NOT present OH as known/observed.
    h2 = _state(operations_health=95.0, _predicted=True)
    assert svc._known_oh_at_cutoff(h2) is None


def test_recursive_state_history_preserved_as_predicted():
    """Predicted recursive states appended to history must never be selected
    as an observed previous-day row."""
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    state = _state(operations_health=80.0, history_buffer=[
        {"operations_health": 99.0, "_predicted": True},
        {"operations_health": 87.0, "_predicted": False},
    ])
    prev = svc._latest_observed_history(state.get("history_buffer"))
    assert prev.get("operations_health") == 87.0  # predicted 99 ignored


# --------------------------------------------------------------------------- #
# enrollment_factor (Task D) — special simulator treatment is narrow
# --------------------------------------------------------------------------- #

def _nps_feature_df(extra_cols=None):
    import pandas as pd
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
    return pd.DataFrame(data)


def test_enrollment_factor_is_cutoff_known_simulator_feature():
    """enrollment_factor is admitted without a companion timestamp because it
    is a deterministic simulator-state feature generated from the enrollment
    state for the observation date itself."""
    from core.nps_predictor.feature_engineering import prepare_features

    df = _nps_feature_df(extra_cols={"enrollment_factor": [1.0, 1.05, 1.1]})
    # Must NOT raise (enrollment_factor needs no _known_at column).
    feats = prepare_features(df)
    assert "enrollment_factor" in feats.columns


def test_enrollment_factor_not_exploitable_for_other_factors():
    """A DIFFERENT unproven factor must NOT slip through the narrow
    enrollment_factor exemption: it requires an explicit _known_at column."""
    from core.nps_predictor.feature_engineering import prepare_features
    import pytest as _p

    # flu_factor is a real external factor; without provenance it must fail.
    df = _nps_feature_df(extra_cols={"flu_factor": [0.5, 0.6, 0.7]})
    with _p.raises(ValueError, match="[Pp]rovenance|_known_at|not known"):
        prepare_features(df)


def test_unproven_external_factor_rejected_without_provenance():
    """A factor with a _known_at column strictly AFTER T must be rejected
    (future information cannot become implicitly valid)."""
    from core.nps_predictor.feature_engineering import prepare_features
    import pytest as _p

    df = _nps_feature_df(extra_cols={
        "flu_factor": [1.0, 1.0, 1.0],
        "flu_factor_known_at": ["2026-01-05", "2026-01-06", "2026-12-31"],
    })
    with _p.raises(ValueError, match="[Pp]rovenance|_known_at|not known"):
        prepare_features(df)


# --------------------------------------------------------------------------- #
# Forecast feature construction — real state propagation (task 7)
# --------------------------------------------------------------------------- #

def test_oh_row_propagates_call_volume_and_oif():
    """Changing the input state must change the constructed OH model row:
    call volume and operational_intelligence_factor are propagated, not
    hardcoded."""
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    base = _state(operations_health=90.0, total_calls_received=1000,
                  operational_intelligence_factor=42)
    row = svc._build_oh_row(base)
    assert row["total_calls_received"] == 1000
    assert row["total_release_calls"] == int(1000 * 60.0 / 100.0)
    assert row["operational_intelligence_factor"] == 42

    changed = _state(operations_health=90.0, total_calls_received=5000,
                     operational_intelligence_factor=99)
    row2 = svc._build_oh_row(changed)
    assert row2["total_calls_received"] == 5000
    assert row2["operational_intelligence_factor"] == 99
    assert row2["total_calls_received"] != row["total_calls_received"]


def test_nps_row_propagates_bif_mif_and_call_volume():
    """BIF / MIF / call volume must come from the input state, not hardcoded 0s."""
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    row = svc._build_nps_row(_state(
        operations_health=90.0,
        business_intelligence_factor=7,
        member_intelligence_factor=8,
        total_calls_received=3000,
    ))
    assert row["business_intelligence_factor"] == 7
    assert row["member_intelligence_factor"] == 8
    assert row["total_calls_received"] == 3000

    # Defaults when absent.
    row2 = svc._build_nps_row(_state(operations_health=90.0))
    assert row2["business_intelligence_factor"] == 0
    assert row2["member_intelligence_factor"] == 0


def test_oh_row_feature_order_stable():
    """The OH feature row keys must be deterministic and stable (order not
    scrambled by propagation changes)."""
    from core.forecast_ai.prediction.service import PredictionService

    svc = PredictionService()
    row = svc._build_oh_row(_state(operations_health=90.0))
    row2 = svc._build_oh_row(_state(operations_health=90.0))
    assert list(row.keys()) == list(row2.keys())
    # The propagated real-state keys are present.
    assert "total_calls_received" in row
    assert "operational_intelligence_factor" in row
