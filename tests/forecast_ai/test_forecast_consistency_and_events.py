"""Focused tests: Forecast output consistency, recursive-history contract,
horizon step counts, and the canonical V2.3 simulator event specification.

The Forecast tests drive ForecastOrchestrator with a stub PredictionService so
the number of prediction steps, the KPI/OH/NPS output source, and the
recursive ``_predicted`` semantics are fully deterministic. The V2.3 tests
verify the canonical event probabilities/effects (absolute points, exactly one
daily event).
"""
from __future__ import annotations

import pytest

from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
from core.forecast_ai.models import ForecastRequest, PredictionResult
from core.forecast_ai.prediction.service import PredictionService
from core.forecast_ai import simulator_events as sev


# ---------------------------------------------------------------------------
# Stub PredictionService with deterministic, controllable KPI output
# ---------------------------------------------------------------------------

class StubPredictionService:
    """Deterministic predictor that records every request and returns fixed
    OH/NPS/KPI values, so the orchestrator's consistency + recursion are
    testable without real (slow) models."""

    def __init__(self, oh=94.0, nps=83.0, quality=90.0, competency=85.0,
                 attendance=88.0, release=55.0, transfer=8.0):
        self.oh = oh
        self.nps = nps
        self.kpis = dict(quality=quality, competency=competency,
                         attendance=attendance, release=release, transfer=transfer)
        self.states = []      # request.state per step
        self.buffers = []     # request.metadata["history_buffer"] per step
        self.days = []        # request.metadata["day"] per call (forecast loop only)

    def predict(self, request):
        meta = request.metadata or {}
        self.states.append(dict(request.state))
        self.buffers.append(list(meta.get("history_buffer", [])))
        self.days.append(meta.get("day"))
        return PredictionResult(
            operations_health=self.oh,
            nps=self.nps,
            quality=self.kpis["quality"],
            competency=self.kpis["competency"],
            attendance=self.kpis["attendance"],
            release=self.kpis["release"],
            transfer=self.kpis["transfer"],
            bayesian_score_distribution=None,
            score_counts=None,
            warnings=[],
            errors=[],
        )

    def forecast_states(self):
        """The forecast-loop prediction states (marked with a day), excluding
        ancillary engine (confidence/risk/sensitivity) model calls."""
        return [s for s, d in zip(self.states, self.days) if d is not None]

    def forecast_buffers(self):
        return [b for b, d in zip(self.buffers, self.days) if d is not None]


def _make_orchestrator(stub):
    return ForecastOrchestrator(prediction_service=stub)


def _request(horizon, state=None):
    state = state or {
        "quality": 85.0, "competency": 78.0, "attendance": 90.0,
        "release": 60.0, "transfer": 9.0, "operations_health": 82.0, "nps": 70.0,
    }
    return ForecastRequest(
        operation="forecast", scenario="baseline", horizon=horizon,
        parameters={"state": dict(state)},
    )


# ---------------------------------------------------------------------------
# Horizon -> prediction-step count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon,expected", [(1, 1), (3, 3), (7, 7)])
def test_horizon_produces_exactly_horizon_prediction_steps(horizon, expected):
    stub = StubPredictionService()
    resp = _make_orchestrator(stub).execute(_request(horizon))
    assert resp.success is True
    assert len(stub.forecast_states()) == expected
    assert len(resp.payload["timeline"]) == expected


# ---------------------------------------------------------------------------
# Recursive predicted-state contract
# ---------------------------------------------------------------------------

def test_recursive_predicted_state_marked_predicted():
    stub = StubPredictionService()
    resp = _make_orchestrator(stub).execute(_request(3))
    assert resp.success is True
    states = stub.forecast_states()
    assert len(states) == 3
    # Day 1 is the observed input (not marked predicted); day 2+ is predicted.
    assert not states[0].get("_predicted")
    assert all(states[i].get("_predicted") is True for i in (1, 2))


def test_recursive_history_buffer_grows_with_predicted_rows():
    stub = StubPredictionService()
    resp = _make_orchestrator(stub).execute(_request(3))
    assert resp.success is True
    buffers = stub.forecast_buffers()
    assert len(buffers) == 3
    # Day 1 has an empty buffer; each later day carries the prior predicted rows.
    assert buffers[0] == []
    assert len(buffers[1]) == 1
    assert len(buffers[2]) == 2
    for buf in buffers[1:]:
        assert all(row.get("_predicted") is True for row in buf)


def test_predicted_history_is_not_treated_as_observed():
    """PredictionService must never read a _predicted row as observed
    known-at-cutoff data, and a predicted state's OH is never fed to NPS."""
    service = PredictionService()
    observed = {"operations_health": 88.0, "_predicted": False,
                "quality": 85.0, "competency": 78.0, "attendance": 90.0,
                "release": 60.0, "transfer": 9.0}
    predicted = dict(observed, operations_health=95.0, _predicted=True)

    # Observed state: its OH is known-at-cutoff.
    assert service._known_oh_at_cutoff(dict(observed)) == 88.0
    # Predicted state: its (predicted) OH is NOT known-at-cutoff.
    assert service._known_oh_at_cutoff(dict(predicted)) is None
    # NPS feature row for a predicted state never receives the predicted OH.
    assert service._build_nps_row(dict(predicted))["operational_health"] is None
    # An all-predicted history buffer is treated as empty (no observed rows).
    buffer = [dict(predicted), dict(predicted)]
    assert service._latest_observed_history(buffer) == {}


# ---------------------------------------------------------------------------
# Forecast KPI/OH/NPS output consistency
# ---------------------------------------------------------------------------

def test_forecast_timeline_kpis_come_from_prediction_result():
    """The timeline KPIs must come from the SAME prediction result as OH/NPS
    (post-prediction), not from the scenario-modified INPUT state."""
    # Input state quality=85; the predictor returns quality=90.
    stub = StubPredictionService(oh=94.0, nps=83.0, quality=90.0)
    resp = _make_orchestrator(stub).execute(_request(1))
    assert resp.success is True
    day = resp.payload["timeline"][0]
    # OH/NPS from the model result.
    assert day["operations_health"] == 94.0
    assert day["nps"] == 83.0
    # KPIs from the same result (post-prediction), NOT the input value (85).
    assert day["quality"] == 90.0
    assert day["competency"] == 85.0
    assert day["attendance"] == 88.0
    assert day["release"] == 55.0
    assert day["transfer"] == 8.0


def test_forecast_timeline_kpis_fallback_when_prediction_lacks_kpis():
    """When the prediction result has no KPI fields (injected-predictor path),
    the timeline falls back to the input state so cells are never None."""
    stub = StubPredictionService()
    # Simulate a result with None KPIs (like the injected-predictor path).
    stub.kpis = dict(quality=None, competency=None, attendance=None,
                     release=None, transfer=None)
    resp = _make_orchestrator(stub).execute(_request(1))
    assert resp.success is True
    day = resp.payload["timeline"][0]
    assert day["quality"] == 85.0      # falls back to input state
    assert day["competency"] == 78.0
    assert day["attendance"] == 90.0


# ---------------------------------------------------------------------------
# Canonical V2.3 simulator event specification
# ---------------------------------------------------------------------------

def test_v23_exactly_eight_events():
    assert set(sev.EVENT_NAMES) == {
        "NORMAL", "PHARMACY_DELAY", "PROVIDER_UPDATE", "CLAIMS_BACKLOG",
        "SYSTEM_SLOWDOWN", "CORE_OUTAGE", "CMS_CHANGE", "TRAINING",
    }
    assert len(sev.EVENT_NAMES) == 8
    assert len(sev.EVENT_EFFECTS) == 8


def test_v23_event_probabilities():
    assert sev.EVENT_PROBABILITIES == {
        "NORMAL": 0.40,
        "PHARMACY_DELAY": 0.10,
        "PROVIDER_UPDATE": 0.20,
        "CLAIMS_BACKLOG": 0.10,
        "SYSTEM_SLOWDOWN": 0.03,
        "CORE_OUTAGE": 0.02,
        "CMS_CHANGE": 0.05,
        "TRAINING": 0.10,
    }


def test_v23_probabilities_sum_to_one():
    assert sev.total_probability() == pytest.approx(1.0)


def test_v23_exactly_one_daily_event():
    """Exactly one mutually-exclusive daily event: each name maps to exactly
    one probability and one effects entry (a categorical single selection)."""
    for name in sev.EVENT_NAMES:
        assert name in sev.EVENT_PROBABILITIES
        assert name in sev.EVENT_EFFECTS
    assert set(sev.EVENT_PROBABILITIES) == set(sev.EVENT_NAMES)
    assert set(sev.EVENT_EFFECTS) == set(sev.EVENT_NAMES)


def test_v23_effects_are_absolute_points():
    """Q/C/R/T/Attendance/OH effects are ABSOLUTE additive points (NORMAL is
    all zeros, non-NORMAL use small integer point deltas), not percentages."""
    assert sev.EVENT_EFFECTS["NORMAL"] == {
        "quality": 0.0, "competency": 0.0, "release": 0.0,
        "transfer": 0.0, "attendance": 0.0, "operations_health": 0.0, "calls": 1.00,
    }
    # Non-NORMAL events carry absolute point effects (integer deltas) on KPIs
    # and a call-volume multiplier.
    for name in ("PHARMACY_DELAY", "PROVIDER_UPDATE", "CLAIMS_BACKLOG",
                 "SYSTEM_SLOWDOWN", "CORE_OUTAGE", "CMS_CHANGE", "TRAINING"):
        eff = sev.EVENT_EFFECTS[name]
        for k in ("quality", "competency", "release", "transfer", "attendance", "operations_health"):
            assert isinstance(eff[k], (int, float))
        assert eff["calls"] > 0.0


@pytest.mark.parametrize("name,effects", [
    ("PHARMACY_DELAY", dict(quality=-2.0, competency=-2.0, release=-2.0,
                            transfer=2.0, attendance=0.0, operations_health=-0.5, calls=1.05)),
    ("PROVIDER_UPDATE", dict(quality=-1.0, competency=-1.0, release=-1.0,
                             transfer=1.0, attendance=0.0, operations_health=-0.3, calls=1.03)),
    ("CLAIMS_BACKLOG", dict(quality=-4.0, competency=-3.0, release=-4.0,
                            transfer=4.0, attendance=0.0, operations_health=-0.8, calls=1.08)),
    ("SYSTEM_SLOWDOWN", dict(quality=-5.0, competency=-2.0, release=-3.0,
                             transfer=4.0, attendance=-2.0, operations_health=-0.6, calls=1.04)),
    ("CORE_OUTAGE", dict(quality=-10.0, competency=-6.0, release=-8.0,
                         transfer=6.0, attendance=-5.0, operations_health=-1.5, calls=1.12)),
    ("CMS_CHANGE", dict(quality=-7.0, competency=-9.0, release=-6.0,
                        transfer=7.0, attendance=-3.0, operations_health=-1.2, calls=1.15)),
    ("TRAINING", dict(quality=3.0, competency=4.0, release=2.0,
                      transfer=-2.0, attendance=0.0, operations_health=0.5, calls=0.90)),
])
def test_v23_event_effect_values(name, effects):
    assert sev.EVENT_EFFECTS[name] == effects
