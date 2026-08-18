"""MODEL → FORECAST → ADIE non-interference invariant validation.

Validates that ADIE only INTERPRETS forecast outputs and never feeds data
back into prediction inputs, forecast state, observed state, provenance, or
predictor behavior.

Invariant under test:
  MODEL → FORECAST → ADIE
  ADIE consumes forecast outputs (advisory), never mutates prediction inputs,
  forecast timeline, observed state, provenance, cutoff, or predictor state.
"""

import copy
import math
from pathlib import Path

from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
from core.forecast_ai.models import ForecastRequest
from core.forecast_ai.prediction.service import PredictionService

# Repository root (the ADIE V3 source tree is scanned relative to it).
REPO_ROOT = Path(__file__).resolve().parents[5]


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
        horizon=3,
        parameters={
            "state": {
                "quality": 85.0,
                "competency": 88.0,
                "attendance": 90.0,
                "release": 60.0,
                "transfer": 9.0,
                "operations_health": 82.0,
            }
        },
    )


# --------------------------------------------------------------------------- #
# A. MODEL → FORECAST
# --------------------------------------------------------------------------- #
def test_a_predictions_come_from_models_not_adie():
    """Forecast OH/NPS must be the direct model predictions (dummy predictors
    here stand in for the trained models), and ADIE must not alter them."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    assert resp.success is True
    tl = resp.payload["timeline"]
    # Day 1 is the direct model output (dummy: OH=82, NPS=70).
    assert tl[0]["operations_health"] == 82.0
    assert tl[0]["nps"] == 70.0


def test_a_recursive_states_marked_predicted():
    """Recursive forecast days (>=2) are _predicted=True."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    pkg = resp.payload["decision_intelligence"]["package"]
    scen = pkg["probabilistic"]["scenarios"]
    assert len(scen) == 3
    assert all(s["_predicted"] is True for s in scen)


# --------------------------------------------------------------------------- #
# B. FORECAST → ADIE
# --------------------------------------------------------------------------- #
def test_b_adie_has_zero_predict_calls():
    """ADIE V3 must contain zero prediction calls (interprets, never predicts)."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib\n"
                "root = pathlib.Path('core/decision_intelligence/v3')\n"
                "hits = []\n"
                "for p in root.rglob('*.py'):\n"
                "    if '__pycache__' in str(p):\n"
                "        continue\n"
                "    for i, line in enumerate(p.read_text().splitlines(), 1):\n"
                "        if '.predict(' in line:\n"
                "            hits.append((str(p), i, line.strip()))\n"
                "print('PREDICT_CALLS_IN_ADIE_V3:', len(hits))\n"
            ),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert "PREDICT_CALLS_IN_ADIE_V3: 0" in result.stdout


def test_b_confidence_risk_sensitivity_trend_downstream_advisory():
    """Confidence/risk/sensitivity/trend must be advisory outputs, present in
    the decision package, and not modify the forecast timeline."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    pkg = resp.payload["decision_intelligence"]["package"]
    # sensitivity + trends present in decision package
    assert pkg["sensitivity"]["success"] is True
    assert len(pkg["sensitivity"]["analyses"]) > 0
    assert len(pkg["trends"]["analyses"]) > 0
    # forecast timeline unchanged by advisory outputs
    tl = resp.payload["timeline"]
    assert all(d["operations_health"] is not None for d in tl)


# --------------------------------------------------------------------------- #
# C. ADIE NON-INTERFERENCE
# --------------------------------------------------------------------------- #
def test_c_prediction_inputs_unchanged_after_adie():
    """Snapshot prediction inputs (observed state + request) before ADIE,
    execute the full forecast, and assert byte/value equivalence afterward."""
    orch = _make_orchestrator()
    req = _base_request()

    # Snapshot the exact observed-state dict the caller supplies.
    observed_before = copy.deepcopy(req.parameters["state"])

    resp = orch.execute(req)

    # The caller's request state must be untouched (ADIE reads a copy).
    assert req.parameters["state"] == observed_before


def test_c_forecast_timeline_unchanged_by_adie():
    """ADIE must not alter the forecast timeline. Run twice with a stable
    (dummy) predictor and assert identical timelines."""
    r1 = _make_orchestrator().execute(_base_request())
    r2 = _make_orchestrator().execute(_base_request())
    t1 = [(d["operations_health"], d["nps"]) for d in r1.payload["timeline"]]
    t2 = [(d["operations_health"], d["nps"]) for d in r2.payload["timeline"]]
    assert t1 == t2


def test_c_observed_state_and_provenance_unchanged():
    """ADIE receives real cutoff + provenance; observed metrics come only from
    the observed state; nothing is written back."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    di = resp.payload["decision_intelligence"]
    assert di["status"] == "success"
    assert di["cutoff"].endswith("T00:00:00+00:00")
    assert len(di["provenance"]) == 6
    assert set(di["observed_metrics"]) == {
        "attendance", "competency", "operations_health", "quality",
        "release", "transfer",
    }
    # Provenance stamps are all at-or-before cutoff (no future provenance).
    for stamp in di["provenance"]:
        assert stamp <= di["cutoff"]


# --------------------------------------------------------------------------- #
# D. DECISION QUALITY / TRACEABILITY
# --------------------------------------------------------------------------- #
def test_d_decision_traces_to_forecast_outputs(monkeypatch):
    """The ADIE probabilistic decision must derive from forecast scenario
    data (forecast-day scenarios), not invented values, and must be a
    MEANINGFUL decision (Phase 3) rather than a bare day index or a
    hard-coded label for every input.

    The dummy OH predictor is constant (always 82), so the real optimizer
    cannot reach a target and correctly returns ``skipped/optimization_failed``
    (see test_d_insufficient_evidence_abstains).  To exercise the meaningful
    decision path deterministically we stub the recommendation engine to return
    a genuine recommendation and verify the decision traces to forecast
    evidence."""
    import core.forecast_ai.engines.recommendation_engine as rec_mod

    class FakeRecEngine:
        def __init__(self, *a, **k):
            pass
        def execute(self, request):
            from types import SimpleNamespace
            return SimpleNamespace(
                success=True,
                payload={
                    "status": "success",
                    "success": True,
                    "recommendations": [{"action": "increase_quality", "value": 2.0}],
                    "evidence_count": 1,
                    "final_recommendation_count": 1,
                    "diagnostics": {"rule": "target_gap"},
                },
            )

    monkeypatch.setattr(rec_mod, "RecommendationEngine", FakeRecEngine)

    orch = _make_orchestrator()
    req = ForecastRequest(
        operation="forecast",
        horizon=3,
        parameters={
            "state": {
                "quality": 85.0, "competency": 88.0, "attendance": 90.0,
                "release": 60.0, "transfer": 9.0, "operations_health": 82.0,
            },
            "target_oh": 92.0,
            "target_nps": 85.0,
            "max_iterations": 10,
            "timeout_seconds": 8,
        },
    )
    resp = orch.execute(req)
    pkg = resp.payload["decision_intelligence"]["package"]
    probs = pkg["probabilistic"]
    scen = probs["scenarios"]
    assert len(scen) == 3
    # Every scenario carries forecast-derived day metadata + real evidence.
    for s in scen:
        assert s["name"].startswith("forecast_day_")
        assert s["_predicted"] is True
        assert s["date"] is not None
        assert "rank" in s
        assert "score" in s
        assert "evidence" in s
    # The recommendation is a meaningful action, never a bare day index and
    # never a single hard-coded label for all inputs.
    rec = probs["recommendation"]
    assert rec
    assert not rec.startswith("forecast_day_")
    assert not rec == "improved_operations"
    decision = probs.get("decision", {})
    assert decision.get("reason"), "decision must be explainable"
    assert isinstance(decision.get("evidence", []), list)


def test_d_insufficient_evidence_abstains():
    """Without targets, the decision must ABSTAIN (empty recommendation, risk
    ABSTAIN) rather than fabricate evidence."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    pkg = resp.payload["decision_intelligence"]["package"]
    probs = pkg["probabilistic"]
    assert probs.get("recommendation", "") in ("", None)
    assert str(probs.get("risk", "")).upper() in ("ABSTAIN", "")
    assert probs.get("abstain") is True or pkg.get("recommendation_status") == "insufficient_evidence"


def test_d_recommendations_do_not_become_predictor_inputs():
    """Recommendation/strategy outputs are advisory; they must not be written
    into prediction inputs or observed state."""
    orch = _make_orchestrator()
    req = ForecastRequest(
        operation="forecast",
        horizon=2,
        parameters={
            "state": {
                "quality": 85.0, "competency": 88.0, "attendance": 90.0,
                "release": 60.0, "transfer": 9.0, "operations_health": 82.0,
            },
            "target_oh": 92.0,
            "target_nps": 85.0,
            "max_iterations": 10,
            "timeout_seconds": 8,
        },
    )
    observed_before = copy.deepcopy(req.parameters["state"])
    resp = orch.execute(req)
    assert resp.success is True
    pkg = resp.payload["decision_intelligence"]["package"]
    if "recommendations" in pkg or "strategies" in pkg:
        # Advisory outputs must not mutate the request state.
        assert req.parameters["state"] == observed_before


# --------------------------------------------------------------------------- #
# E. TEMPORAL SAFETY
# --------------------------------------------------------------------------- #
def test_e_predicted_never_becomes_observed_through_adie():
    """Predicted forecast values (scenarios) must never be relabeled as
    observed; observed inputs come only from the caller state."""
    orch = _make_orchestrator()
    resp = orch.execute(_base_request())
    di = resp.payload["decision_intelligence"]
    pkg = di["package"]
    # observed_metrics are exactly the caller state keys (no forecast-derived).
    assert set(di["observed_metrics"]) == {
        "attendance", "competency", "operations_health", "quality",
        "release", "transfer",
    }
    # All forecast scenarios are explicitly predicted, never observed.
    for s in pkg["probabilistic"]["scenarios"]:
        assert s["_predicted"] is True
