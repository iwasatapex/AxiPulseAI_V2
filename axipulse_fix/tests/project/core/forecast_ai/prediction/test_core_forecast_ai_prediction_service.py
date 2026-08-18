import importlib


def test_recursive_prediction_state_does_not_carry_prior_predicted_nps_as_input():
    from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
    from core.forecast_ai.models import ForecastRequest
    from core.forecast_ai.prediction.service import PredictionService

    class DummyOHPredictor:
        def predict(self, state):
            return 80.0

    class RecordingNPSPredictor:
        def __init__(self):
            self.seen_states = []

        def predict(self, state):
            self.seen_states.append(dict(state))
            return {"nps": 70.0}

    service = PredictionService(
        oh_predictor=DummyOHPredictor(),
        nps_predictor=RecordingNPSPredictor(),
    )
    orchestrator = ForecastOrchestrator(prediction_service=service)

    request = ForecastRequest(
        operation="forecast",
        horizon=2,
        parameters={
            "state": {
                "quality": 85.0,
                "competency": 78.0,
                "release": 60.0,
                "transfer": 9.0,
                "attendance": 90.0,
            }
        },
    )

    result = orchestrator.execute(request)

    assert result.success is True
    seen_states = service.nps.seen_states
    assert len(seen_states) == 2
    assert "nps" not in seen_states[0]
    assert "nps" not in seen_states[1]


def test_predicted_history_buffer_is_not_treated_as_observed_previous_day_state():
    from core.forecast_ai.prediction.service import PredictionService

    service = PredictionService()
    state = {
        "quality": 85.0,
        "competency": 78.0,
        "release": 60.0,
        "transfer": 9.0,
        "attendance": 90.0,
        "history_buffer": [
            {
                "quality": 50.0,
                "competency": 51.0,
                "release": 52.0,
                "transfer": 53.0,
                "attendance": 54.0,
                "operations_health": 55.0,
                "nps": 56.0,
                "_predicted": True,
            }
        ],
    }

    oh_row = service._build_oh_row(state)
    nps_row = service._build_nps_row(state)

    assert oh_row["quality_previous_day"] == state["quality"]
    assert oh_row["competency_previous_day"] == state["competency"]
    assert oh_row["release_previous_day"] == state["release"]
    assert oh_row["transfer_previous_day"] == state["transfer"]
    assert oh_row["attendance_previous_day"] == state["attendance"]
    assert oh_row["operations_health_previous_day"] == 80

    assert nps_row["quality_previous_day"] == state["quality"]
    assert nps_row["competency_previous_day"] == state["competency"]
    assert nps_row["release_previous_day"] == state["release"]
    assert nps_row["transfer_previous_day"] == state["transfer"]
    assert nps_row["attendance_previous_day"] == state["attendance"]
    assert nps_row["nps_previous_day"] == 0
    assert nps_row["operational_health"] is None

def test_predicted_operations_health_in_state_is_not_known_at_cutoff_for_nps():
    """Regression: predicted OH carried in recursive state must never become
    the known-at-cutoff OH consumed by the NPS feature row.

    This tests the DIRECT-STATE path (predicted operations_health present in
    the state dict itself with the _predicted marker), not only the
    history-buffer row path.
    """
    from core.forecast_ai.prediction.service import PredictionService

    service = PredictionService()
    state = {
        "quality": 85.0,
        "competency": 78.0,
        "release": 60.0,
        "transfer": 9.0,
        "attendance": 90.0,
        "operations_health": 55.0,  # predicted recursive OH
        "_predicted": True,          # recursive predicted state marker
        "history_buffer": [],
    }

    nps_row = service._build_nps_row(state)

    assert nps_row["operational_health"] is None
    assert nps_row["operational_health"] != 55.0


def test_predicted_operations_health_in_state_is_not_previous_observed_oh():
    """Regression: predicted OH in recursive state must not become the
    OH previous-day input of the OH feature row.
    """
    from core.forecast_ai.prediction.service import PredictionService

    service = PredictionService()
    state = {
        "quality": 85.0,
        "competency": 78.0,
        "release": 60.0,
        "transfer": 9.0,
        "attendance": 90.0,
        "operations_health": 55.0,  # predicted recursive OH
        "_predicted": True,          # recursive predicted state marker
        "history_buffer": [],
    }

    oh_row = service._build_oh_row(state)

    assert oh_row["operations_health_previous_day"] != 55.0


def test_observed_operations_health_in_state_is_still_known_at_cutoff():
    """Sanity: an OBSERVED state (no _predicted marker) with a known OH value
    still feeds the NPS known-at-cutoff feature. The fix must not break the
    legitimate observed path.
    """
    from core.forecast_ai.prediction.service import PredictionService

    service = PredictionService()
    state = {
        "quality": 85.0,
        "competency": 78.0,
        "release": 60.0,
        "transfer": 9.0,
        "attendance": 90.0,
        "operations_health": 90.0,  # observed known-at-cutoff OH
        "history_buffer": [],
    }

    nps_row = service._build_nps_row(state)
    oh_row = service._build_oh_row(state)

    assert nps_row["operational_health"] == 90.0
    assert oh_row["operations_health_previous_day"] == 90.0


def test_recursive_state_with_mixed_history_uses_only_observed_oh():
    """Regression: even when the state carries predicted OH, if observed
    history exists the observed OH (not the predicted one) must win.
    """
    from core.forecast_ai.prediction.service import PredictionService

    service = PredictionService()
    state = {
        "quality": 85.0,
        "competency": 78.0,
        "release": 60.0,
        "transfer": 9.0,
        "attendance": 90.0,
        "operations_health": 55.0,  # predicted recursive OH
        "_predicted": True,
        "history_buffer": [
            {
                "quality": 80.0,
                "competency": 75.0,
                "release": 58.0,
                "transfer": 10.0,
                "attendance": 89.0,
                "operations_health": 88.0,  # observed
                "nps": 70.0,
                # NOTE: no _predicted marker -> observed row
            }
        ],
    }

    nps_row = service._build_nps_row(state)

    assert nps_row["operational_health"] == 88.0
    assert nps_row["operational_health"] != 55.0


def test_orchestrator_marks_recursive_state_predicted():
    """Regression: ForecastOrchestrator must mark recursive predicted state
    dicts (day >= 2) with _predicted=True so PredictionService guards trigger.
    """
    from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
    from core.forecast_ai.models import ForecastRequest
    from core.forecast_ai.prediction.service import PredictionService

    seen = {}

    class DummyOHPredictor:
        def predict(self, state):
            return 80.0

    class RecordingNPSPredictor:
        def __init__(self):
            self.seen_states = []

        def predict(self, state):
            self.seen_states.append(dict(state))
            return {"nps": 70.0}

    service = PredictionService(
        oh_predictor=DummyOHPredictor(),
        nps_predictor=RecordingNPSPredictor(),
    )
    orchestrator = ForecastOrchestrator(prediction_service=service)

    request = ForecastRequest(
        operation="forecast",
        horizon=2,
        parameters={
            "state": {
                "quality": 85.0,
                "competency": 78.0,
                "release": 60.0,
                "transfer": 9.0,
                "attendance": 90.0,
            }
        },
    )

    result = orchestrator.execute(request)

    assert result.success is True
    seen_states = service.nps.seen_states
    assert len(seen_states) == 2
    # Day 1 state is observed (not predicted); day 2 state is recursive/predicted.
    assert seen_states[0].get("_predicted") is not True
    assert seen_states[1].get("_predicted") is True
    seen["ok"] = True


def test_service_surface():
    module = importlib.import_module("core.forecast_ai.prediction.service")
    assert hasattr(module, "predict")
    assert hasattr(module, "predict_oh")
    assert hasattr(module, "predict_nps")
    assert hasattr(module, "PredictionService")
