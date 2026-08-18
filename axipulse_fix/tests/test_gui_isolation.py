"""Adversarial session-isolation and concurrency regression tests.

These exercise the *real* isolation mechanisms the GUI relies on:

* ``gui.state`` stores session state per Streamlit session (simulated here
  by routing the store to per-"session" dicts — the exact abstraction the
  GUI uses, since Streamlit provides one session-state store per browser
  session).
* ``gui.services`` activates the process-global ``PredictorProvider`` under
  a lock using the *explicit* family for each request, so concurrent
  requests can never cross model families.

The provider-activation race is NOT mocked away: tests use the real
``PredictorProvider.set_model_family`` and the real ``_PROVIDER_LOCK``.
Only the expensive pipeline/orchestrator execution is stubbed (it is the
model-running step, not the selection/isolation step under test).
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gui import services as svc


def _valid_state():
    return {
        "quality": 87.0, "competency": 93.0, "attendance": 90.0,
        "release": 60.0, "transfer": 9.0, "operations_health": 95.0,
        "nps": 82.0, "total_calls_received": 2000.0,
    }


def _pipeline_result():
    """Minimal ProductionPredictionResult-shaped object for _prediction_to_dict."""
    raw = SimpleNamespace(operations_health=None, nps=None)
    return SimpleNamespace(
        prediction=SimpleNamespace(
            raw=raw, operations_health=None, nps=None,
            bayesian_score_distribution=None, score_counts=None,
        )
    )


def _run_two_sessions(monkeypatch):
    """Route gui.state's store to per-session dicts (the real abstraction)."""
    from gui import state as gui_state

    stores = {"A": {}, "B": {}}
    current = ["A"]

    def fake_store():
        return stores[current[0]]

    monkeypatch.setattr(gui_state, "_store", fake_store)
    return stores, current


# =====================================================================
# 1. Two-session model isolation (via the session-store abstraction)
# =====================================================================

def test_two_sessions_hold_different_families(monkeypatch):
    from gui.state import GUIState

    stores, current = _run_two_sessions(monkeypatch)
    st = GUIState()

    # Session A selects alpha.
    current[0] = "A"
    st.set_active_family("alpha")
    # Session B selects beta.
    current[0] = "B"
    st.set_active_family("beta")

    # A still alpha, B still beta — no leakage.
    current[0] = "A"
    assert st.get_active_family() == "alpha"
    current[0] = "B"
    assert st.get_active_family() == "beta"

    # Selecting in A does not alter B, and vice versa.
    current[0] = "A"
    st.set_active_family("alpha2")
    current[0] = "B"
    assert st.get_active_family() == "beta"
    current[0] = "B"
    st.set_active_family("beta2")
    current[0] = "A"
    assert st.get_active_family() == "alpha2"


def test_session_selection_does_not_mutate_other_sessions_active_family(monkeypatch):
    stores, current = _run_two_sessions(monkeypatch)
    svc.STATE.reset()

    current[0] = "A"
    svc.STATE.set_active_family("alpha")
    current[0] = "B"
    svc.STATE.set_active_family("beta")

    current[0] = "A"
    assert svc.STATE.get_active_family() == "alpha"
    current[0] = "B"
    assert svc.STATE.get_active_family() == "beta"
    svc.STATE.reset()


# =====================================================================
# 2. Concurrent provider test (real lock + real provider activation)
# =====================================================================

@pytest.mark.parametrize("iterations", [25, 75])
def test_concurrent_provider_never_crosses_families(monkeypatch, iterations):
    """Two workers hammer the provider with different families.

    Each request uses the real PredictorProvider.set_model_family under the
    real _PROVIDER_LOCK.  If the lock were mis-scoped, a worker could read
    the other worker's family mid-request and the invariant
    "A never sees B / B never sees A" would break.
    """
    failures = []
    a_runs = 0
    b_runs = 0
    counter_lock = threading.Lock()

    class StubPipeline:
        def run(self, state):
            # The family active on the provider DURING this request.
            got = svc.PredictorProvider.get_model_family()
            expected = state.get("_expected_family")
            if got != expected:
                with counter_lock:
                    failures.append(
                        f"expected provider={expected!r} but got {got!r}"
                    )
            return _pipeline_result()

    monkeypatch.setattr(
        "core.forecast_ai.prediction.pipeline.ProductionPredictionPipeline",
        StubPipeline,
    )
    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))

    def worker(fam, n):
        nonlocal a_runs, b_runs
        for _ in range(n):
            state = _valid_state()
            state["_expected_family"] = fam
            svc.predict(state, family=fam)
            with counter_lock:
                if fam == "alpha":
                    a_runs += 1
                else:
                    b_runs += 1

    ta = threading.Thread(target=worker, args=("alpha", iterations))
    tb = threading.Thread(target=worker, args=("beta", iterations))
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    assert not failures, "provider family crossed between concurrent requests: " + "; ".join(failures)
    assert a_runs == iterations and b_runs == iterations
    svc.STATE.reset()


def test_concurrent_predict_active_family_in_payload(monkeypatch):
    """Returned payload's active_family always equals the requesting family."""
    seen = []
    seen_lock = threading.Lock()

    class StubPipeline:
        def run(self, state):
            return _pipeline_result()

    monkeypatch.setattr(
        "core.forecast_ai.prediction.pipeline.ProductionPredictionPipeline",
        StubPipeline,
    )
    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))

    def worker(fam, n):
        for _ in range(n):
            out = svc.predict(_valid_state(), family=fam)
            with seen_lock:
                seen.append(out["active_family"])

    ts = [threading.Thread(target=worker, args=(f, 20)) for f in ("alpha", "beta", "gamma")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert seen.count("alpha") == 20
    assert seen.count("beta") == 20
    assert seen.count("gamma") == 20
    svc.STATE.reset()


# =====================================================================
# 3 & 4. Training completion ownership (session-scoped activation)
# =====================================================================

def test_train_models_does_not_activate_any_session(monkeypatch):
    """train_models must NOT touch session state (activation is the view's job)."""
    import tempfile
    from pathlib import Path
    import core.nps_predictor.predictor as nps_mod
    import core.operation_health_predictor.predictor as oh_mod

    stores, current = _run_two_sessions(monkeypatch)
    svc.STATE.reset()
    current[0] = "A"
    svc.STATE.set_active_family("alpha")

    class StubOH:
        model_name = "CatBoost"
        feature_names = ["a"]
        algorithm_performance = {}
        history_days = 1
        def train(self, path): pass
        def save_model(self, path): pass

    class StubNPS:
        model_name = "RF"
        feature_names = ["x"]
        algorithm_performance = {}
        history_days = 1
        def train(self, path): pass
        def save_model(self, path): pass

    monkeypatch.setattr(oh_mod, "OperationalHealthPredictor", StubOH)
    monkeypatch.setattr(nps_mod, "NPSPredictor", StubNPS)
    td = Path(tempfile.mkdtemp())
    f = td / "gamma.csv"
    f.write_text(
        "actual_quality,actual_competency,actual_attendance,"
        "actual_release_rate,actual_transfer_rate,promoters,passives,detractors\n"
        "80,70,85,55,5,10,20,30\n"
    )
    monkeypatch.setattr(svc, "list_training_files", lambda: [f])
    monkeypatch.setattr(svc, "MODELS_DIR", td / "models")

    out = svc.train_models("gamma.csv")

    # train_models returns the family but must NOT set active family anywhere.
    assert out["family"] == "gamma"
    # Session A is still on alpha after training completes (nothing mutated).
    current[0] = "A"
    assert svc.STATE.get_active_family() == "alpha"
    current[0] = "B"
    assert svc.STATE.get_active_family() is None
    svc.STATE.reset()


def test_train_completion_activation_only_in_initiating_session(monkeypatch):
    """Simulates train_view's main-thread completion: select_model_family
    only changes the initiating session's selection."""
    from gui.state import GUIState

    stores, current = _run_two_sessions(monkeypatch)
    st = GUIState()

    current[0] = "A"
    st.set_active_family("alpha")
    current[0] = "B"
    st.set_active_family("beta")

    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))

    # Session A's Train view completes and activates family C (main thread).
    current[0] = "A"
    svc.select_model_family("gamma")

    current[0] = "A"
    assert st.get_active_family() == "gamma"
    current[0] = "B"
    assert st.get_active_family() == "beta"  # B untouched by A's training
    svc.STATE.reset()
