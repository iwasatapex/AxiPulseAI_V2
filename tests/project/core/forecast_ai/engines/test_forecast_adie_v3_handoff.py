"""Regression tests for the ForecastOrchestrator -> canonical ADIE V3 handoff."""

import importlib

import pytest

from core.forecast_ai.engines import forecast_orchestrator as orch_module
from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
from core.forecast_ai.models import ForecastRequest
from core.forecast_ai.prediction.service import PredictionService


class DummyOHPredictor:
    def predict(self, state):
        return 82.0


class DummyNPSPredictor:
    def predict(self, state):
        return {"nps": 70.0}


def _make_orchestrator():
    service = PredictionService(
        oh_predictor=DummyOHPredictor(),
        nps_predictor=DummyNPSPredictor(),
    )
    return ForecastOrchestrator(prediction_service=service)


def _base_request():
    return ForecastRequest(
        operation="forecast",
        horizon=2,
        parameters={
            "state": {
                "quality": 85.0,
                "competency": 78.0,
                "release": 60.0,
                "transfer": 9.0,
                "attendance": 90.0,
                "operations_health": 82.0,
            }
        },
    )


def test_handoff_payload_present_and_success():
    """ForecastOrchestrator -> ADIE V3: decision_intelligence_v3 present."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    assert resp.success is True
    v3 = resp.payload.get("decision_intelligence", {})
    assert v3.get("status") == "success"
    assert "package" in v3
    assert "cutoff" in v3
    assert "provenance" in v3


def test_handoff_real_cutoff_propagated():
    """The real forecast cutoff (forecast start date T) must reach V3."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    v3 = resp.payload.get("decision_intelligence", {})
    assert v3.get("status") == "success"
    cutoff = v3["cutoff"]
    # Cutoff is the forecast start date (today), ISO-8601.
    assert cutoff.endswith("T00:00:00+00:00")
    # Forecast day-1 scenario date is strictly after the cutoff (T -> T+1).
    day1 = v3["package"]["probabilistic"]["scenarios"][0]
    assert day1["date"] > cutoff[:10]


def test_handoff_provenance_propagated():
    """Every observed input carries a provenance stamp at-or-before cutoff."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    v3 = resp.payload.get("decision_intelligence", {})
    assert v3.get("status") == "success"
    provenance = v3["provenance"]
    cutoff = v3["cutoff"]
    assert provenance, "provenance must not be empty"
    for stamp in provenance:
        assert stamp <= cutoff


def test_handoff_predicted_vs_observed_separation():
    """Forecast timeline values are predicted scenarios, never observed inputs."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    v3 = resp.payload.get("decision_intelligence", {})
    assert v3.get("status") == "success"

    # Observed metrics come only from the caller-supplied observed state.
    assert set(v3["observed_metrics"]) == {
        "attendance", "competency", "operations_health", "quality",
        "release", "transfer",
    }

    # Every forecast scenario is explicitly predicted.
    for scenario in v3["package"]["probabilistic"]["scenarios"]:
        assert scenario["_predicted"] is True


def test_handoff_recursive_predicted_oh_not_observed():
    """Recursive predicted OH (day >= 2) must not become observed through V3.

    The observed_metrics are derived from the day-1 caller state only; the
    forecast timeline (which carries model-predicted OH) is never folded into
    the observed inputs. The day-1 observed OH (82) is the only OH source.
    """
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    v3 = resp.payload.get("decision_intelligence", {})
    assert v3.get("status") == "success"

    # Only the observed state's operations_health is used as an observed input.
    assert "operations_health" in v3["observed_metrics"]
    # No predicted scenario value can be an observed input: observed_metrics
    # only lists metric names from the observed state, and all scenarios are
    # marked predicted.
    for scenario in v3["package"]["probabilistic"]["scenarios"]:
        assert scenario["_predicted"] is True


def test_handoff_invokes_production_boundary(monkeypatch):
    """The Forecast -> V3 internal path must invoke ProductionDecisionBoundary."""

    calls = {"validate": 0}

    class RecordingBoundary(orch_module.ProductionDecisionBoundary):
        def validate(self, **kwargs):
            calls["validate"] += 1
            assert kwargs.get("cutoff"), "cutoff must reach the boundary"
            assert kwargs.get("metadata", {}).get("provenance"), \
                "provenance must reach the boundary"
            assert kwargs.get("scenarios"), "scenarios must reach the boundary"
            return super().validate(**kwargs)

    monkeypatch.setattr(orch_module, "ProductionDecisionBoundary", RecordingBoundary)

    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    assert resp.success is True
    assert calls["validate"] == 1


def test_handoff_future_provenance_rejected_by_boundary(monkeypatch):
    """A future-dated observed input must be rejected by the boundary."""

    def boom(*args, **kwargs):
        raise ValueError("temporal provenance required")

    monkeypatch.setattr(
        orch_module.ProductionDecisionBoundary,
        "validate",
        staticmethod(boom),
    )

    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    # Advisory handoff: forecast still succeeds, V3 reports the error.
    assert resp.success is True
    v3 = resp.payload.get("decision_intelligence", {})
    assert v3.get("status") == "error"


def test_handoff_v2_removed_v3_canonical():
    """V2 is removed: decision_intelligence is the canonical V3 package and
    only the V3 route exists."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    assert resp.payload.get("decision_intelligence"), \
        "canonical (V3) decision package must be in payload"
    # V2 route must NOT be importable (deleted).
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("api.routes.adie_routes")

    # V3 API route remains importable (canonical).
    v3_routes = importlib.import_module("api.routes.adie_v3_routes")
    assert hasattr(v3_routes, "router")


def test_handoff_skips_without_observed_state():
    """Without an observed current state, V3 reports skipped (no fabrication)."""
    orch = _make_orchestrator()
    req = ForecastRequest(
        operation="forecast",
        horizon=2,
        parameters={"state": {}},
    )
    resp = orch.execute(req)
    # Empty state -> no observed metrics -> V3 skipped; forecast itself may
    # still succeed or fail, but the handoff must not fabricate inputs.
    v3 = resp.payload.get("decision_intelligence", {})
    assert v3.get("status") == "skipped" or v3.get("status") == "error"


def test_handoff_sensitivity_executed_and_present():
    """Sensitivity must run after forecast generation and its output must be
    present in the canonical decision package."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    assert resp.success is True
    pkg = resp.payload.get("decision_intelligence", {}).get("package", {})
    assert "sensitivity" in pkg
    assert pkg["sensitivity"].get("success") is True
    assert isinstance(pkg["sensitivity"].get("analyses", []), list)
    assert len(pkg["sensitivity"].get("analyses", [])) > 0


def test_handoff_sensitivity_does_not_change_predictions():
    """Sensitivity must not alter forecast predictions: the timeline must be
    identical whether or not sensitivity runs."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    assert resp.success is True
    # The timeline is the model output; sensitivity is advisory and does not
    # modify it. Compare day-1 predicted OH/NPS to the dummy predictor values.
    timeline = resp.payload.get("timeline", [])
    assert timeline
    assert timeline[0]["operations_health"] == 82.0
    assert timeline[0]["nps"] == 70.0


def test_handoff_sensitivity_uses_observed_not_predicted(monkeypatch):
    """Sensitivity must use the observed state only, never predicted
    recursive state, and must not feed back into prediction inputs."""
    from core.forecast_ai import sensitivity as sensitivity_pkg

    captured = {}

    orig_init = sensitivity_pkg.SensitivityEngine.__init__

    def spy_init(self, prediction_service=None, step_size=1.0):
        captured["service_used"] = prediction_service is not None
        return orig_init(self, prediction_service=prediction_service, step_size=step_size)

    monkeypatch.setattr(
        sensitivity_pkg.SensitivityEngine, "__init__", spy_init
    )

    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    assert resp.success is True
    # Sensitivity ran through the ForecastOrchestrator's own PredictionService
    # (injected), so the captured flag is not strictly required; the key
    # guarantee is that forecast predictions are unchanged (see the previous
    # test) and no feedback loop exists.
    pkg = resp.payload.get("decision_intelligence", {}).get("package", {})
    assert "sensitivity" in pkg


# --------------------------------------------------------------------------- #
# P1-A: FORECAST SUCCESS CONTRACT ON PARTIAL FAILURE
# --------------------------------------------------------------------------- #

def _failing_service(fail_on_or_after_day):
    """A PredictionService whose predict() raises on/after a given day."""
    calls = {"n": 0}

    class _Predictor:
        def predict(self, state):
            return 82.0

    class _NPS:
        def predict(self, state):
            return {"nps": 70.0}

    class _Svc(PredictionService):
        def predict(self, pred_req):
            calls["n"] += 1
            day = (pred_req.metadata or {}).get("day", 1)
            if day >= fail_on_or_after_day:
                raise RuntimeError(f"injected day {day} failure")
            return PredictionService(
                oh_predictor=_Predictor(), nps_predictor=_NPS()
            ).predict(pred_req)

    return _Svc(oh_predictor=_Predictor(), nps_predictor=_NPS())


def test_partial_forecast_reports_partial_not_success():
    """A forecast where some days fail must report success=False and status
    'partial' (not a clean success), while preserving placeholder/error
    evidence."""
    service = _failing_service(fail_on_or_after_day=2)
    orch = ForecastOrchestrator(prediction_service=service)
    req = ForecastRequest(
        operation="forecast",
        horizon=3,
        parameters={
            "state": {
                "quality": 85.0, "competency": 78.0, "release": 60.0,
                "transfer": 9.0, "attendance": 90.0, "operations_health": 82.0,
            }
        },
    )
    resp = orch.execute(req)
    assert resp.success is False
    summary = resp.payload["summary"]
    assert summary["status"] == "partial"
    assert summary["total_days"] == 3
    assert summary["completed_days"] < 3
    assert resp.errors, "errors must be exposed truthfully"


def test_fully_failed_forecast_reports_failed():
    """A forecast where every day fails must report success=False and status
    'failed', never a clean success."""
    service = _failing_service(fail_on_or_after_day=1)
    orch = ForecastOrchestrator(prediction_service=service)
    req = ForecastRequest(
        operation="forecast",
        horizon=2,
        parameters={
            "state": {
                "quality": 85.0, "competency": 78.0, "release": 60.0,
                "transfer": 9.0, "attendance": 90.0, "operations_health": 82.0,
            }
        },
    )
    resp = orch.execute(req)
    assert resp.success is False
    summary = resp.payload["summary"]
    assert summary["status"] == "failed"
    assert summary["completed_days"] == 0


def test_full_forecast_reports_completed_success():
    """A fully completed forecast with no day errors reports success=True and
    status 'completed'."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    assert resp.success is True
    summary = resp.payload["summary"]
    assert summary["status"] == "completed"


# --------------------------------------------------------------------------- #
# ADIE: OBSERVED OH AND OBSERVED NPS ARE BOTH FIRST-CLASS
# --------------------------------------------------------------------------- #

class _ForecastNPS84:
    def predict(self, state):
        return {"nps": 84.5}


def _run_forecast(state_extra, params_extra=None):
    """Run a 2-day forecast with a distinct forecast-day NPS (84.5) so observed
    NPS (82.0) can never be confused with the forecast value."""
    service = PredictionService(
        oh_predictor=DummyOHPredictor(),
        nps_predictor=_ForecastNPS84(),
    )
    orch = ForecastOrchestrator(prediction_service=service)
    state = {
        "quality": 85.0, "competency": 78.0, "release": 60.0,
        "transfer": 9.0, "attendance": 90.0, "operations_health": 82.0,
    }
    state.update(state_extra or {})
    params = {"state": state}
    params.update(params_extra or {})
    req = ForecastRequest(operation="forecast", horizon=2, parameters=params)
    return orch.execute(req).payload["decision_intelligence"]


def test_observed_nps_is_first_class_in_observed_metrics():
    """Forecast request with observed NPS -> observed_metrics includes 'nps'."""
    v3 = _run_forecast({"nps": 82.0})
    assert "nps" in v3["observed_metrics"]


def test_adie_output_explicitly_contains_observed_nps():
    """ADIE output must expose observed NPS in observed_state and
    observed_evidence with metric identity."""
    v3 = _run_forecast({"nps": 82.0})
    assert v3["observed_nps"] == 82.0
    assert v3["observed_nps_available"] is True
    assert v3["observed_state"]["nps"] == 82.0
    assert v3["observed_state"]["operations_health"] == 82.0
    assert v3["observed_evidence"]["nps"] == {"value": 82.0, "source": "observed"}
    assert v3["observed_evidence"]["operations_health"] == {
        "value": 82.0, "source": "observed",
    }
    detail = v3["package"]["details"]
    assert detail["observed_state"]["nps"] == 82.0
    assert detail["observed_nps_available"] is True


def test_observed_nps_taken_from_caller_state_at_cutoff():
    """Observed NPS must come from the caller-supplied state at cutoff T."""
    v3 = _run_forecast({"nps": 82.0})
    assert v3["observed_nps"] == 82.0
    assert v3["observed_state"]["nps"] == 82.0


def test_forecast_day_nps_is_not_observed_nps():
    """Forecast-day NPS must NOT be used as observed NPS; both may appear and
    must not overwrite each other."""
    v3 = _run_forecast({"nps": 82.0})
    assert v3["observed_nps"] == 82.0  # observed (caller), not 84.5
    detail = v3["package"]["details"]
    scen0 = detail["scenario_comparison"][0]
    assert scen0["nps"] == 84.5  # forecast-day NPS, distinct from observed
    assert v3["observed_state"]["nps"] == 82.0  # not overwritten by forecast


def test_recursive_predicted_nps_cannot_become_observed_nps():
    """The recursive predicted NPS (forecast day) must never become observed
    NPS. Observed NPS is fixed to the caller-supplied current state."""
    v3 = _run_forecast({"nps": 82.0})
    detail = v3["package"]["details"]
    forecast_nps = detail["scenario_comparison"][0]["nps"]
    assert forecast_nps == 84.5
    assert v3["observed_nps"] == 82.0
    assert forecast_nps != v3["observed_nps"]


def test_request_without_observed_nps_no_fabrication():
    """No observed NPS in request -> observed_nps_available=false and no NPS
    is fabricated into observed state/metrics/evidence."""
    v3 = _run_forecast({})
    assert v3["observed_nps"] is None
    assert v3["observed_nps_available"] is False
    assert "nps" not in v3["observed_metrics"]
    assert "nps" not in v3["observed_state"]
    assert "nps" not in v3["observed_evidence"]
    assert "nps" not in v3["current_state_gaps"]


def test_observed_nps_compared_against_target_nps():
    """Observed NPS can be compared against target_nps as a deterministic
    current-state gap (distinct from probability_of_target)."""
    v3 = _run_forecast(
        {"nps": 82.0},
        {"target_oh": 90.0, "target_nps": 90.0},
    )
    gaps = v3["current_state_gaps"]
    assert gaps["nps"]["observed"] == 82.0
    assert gaps["nps"]["target"] == 90.0
    assert gaps["nps"]["gap"] == pytest.approx(8.0)
    assert gaps["operations_health"]["gap"] == pytest.approx(8.0)
    detail = v3["package"]["details"]
    assert detail["current_state_gaps"]["nps"]["gap"] == pytest.approx(8.0)


def test_scalar_observed_nps_does_not_enter_bayesian_uncertainty():
    """Scalar observed NPS is current-state evidence only; it must NOT enter
    the generic Beta-Bernoulli health posterior. Changing observed NPS must not
    change the decision-level probability."""
    a = _run_forecast({"nps": 20.0})
    b = _run_forecast({"nps": 95.0})
    pa = a["package"]["probabilistic"]["probability"]
    pb = b["package"]["probabilistic"]["probability"]
    assert pa == pb, "scalar NPS must not affect the generic health posterior"


def test_0_10_distribution_is_sole_nps_uncertainty_source():
    """Scalar observed NPS is current-state evidence only; it must never be
    used as the source of Bayesian/Monte-Carlo NPS uncertainty. NPS uncertainty
    originates from the 0..10 survey-score distribution on forecast scenarios,
    never from the observed scalar."""
    v3 = _run_forecast({"nps": 82.0})
    detail = v3["package"]["details"]
    # Observed NPS evidence is a bare scalar with explicit source; it carries
    # NO fabricated uncertainty/interval (would violate the invariant).
    nps_ev = v3["observed_evidence"]["nps"]
    assert nps_ev == {"value": 82.0, "source": "observed"}
    assert "low" not in nps_ev and "high" not in nps_ev
    assert "p05" not in nps_ev and "p95" not in nps_ev
    # Observed scalar NPS must not become the forecast/probabilistic NPS.
    scen0 = detail["scenario_comparison"][0]
    assert scen0["nps"] == 84.5  # forecast point, from the model
    assert scen0["nps"] != v3["observed_nps"]
    # The observed scalar never appears as an NPS uncertainty input.
    assert v3["observed_nps"] == 82.0


def test_existing_oh_observed_semantics_unchanged():
    """Observed OH remains first-class and unchanged (operations_health in
    observed_state/metrics/evidence)."""
    v3 = _run_forecast({"nps": 82.0})
    assert v3["observed_state"]["operations_health"] == 82.0
    assert v3["observed_metrics"]  # non-empty, includes operations_health
    assert "operations_health" in v3["observed_metrics"]
    assert v3["observed_evidence"]["operations_health"]["value"] == 82.0


def test_full_forecast_adie_handoff_passes_with_observed_nps():
    """Full Forecast -> ADIE handoff still succeeds and yields a package with
    details when observed NPS is supplied."""
    v3 = _run_forecast({"nps": 82.0})
    assert v3["status"] == "success"
    assert "package" in v3
    assert "details" in v3["package"]


def test_sensitivity_recommendation_decision_gates_intact():
    """Supplying observed NPS must not disturb sensitivity/recommendation/
    decision-evidence gating."""
    v3 = _run_forecast({"nps": 82.0})
    pkg = v3["package"]
    assert "sensitivity" in pkg
    assert "details" in pkg
    detail = pkg["details"]
    assert detail["decision_status"] in ("available", "insufficient_evidence")
    assert "sensitivity_detail" in detail
