"""Regression tests for the Reverse Optimizer GUI *state lifecycle*.

These prove the fix for the stale-GUI-state bug: the objective checkboxes are
authoritative, inactive objectives are treated as ``None`` and never forwarded
to the canonical ReverseOptimizer, at least one objective is required before
running, and a previously-rendered ``reverse_result`` is invalidated the moment
the objective configuration (family / which objectives / target values) changes.

Coverage:
  * pure helpers (``objective_signature`` / ``active_targets`` /
    ``current_reverse_result``) drive the deterministic staleness decision;
  * end-to-end Streamlit ``AppTest`` exercises the REAL ``reverse_view.render()``
    UI (checkbox/target/family changes, stale-result invalidation, no-op when no
    objective is selected) against the canonical ``reverse_optimize_canonical``;
  * service-level checks that joint candidates keep predicted OH/NPS and the
    canonical Monte-Carlo NPS interval.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

from core.forecast_ai.models import ForecastResponse
from gui import services as svc
from gui.state import STATE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.views.reverse_view import (  # noqa: E402
    REVERSE_RESULT_KEY,
    REVERSE_RESULT_SIGNATURE_KEY,
    active_targets,
    current_reverse_result,
    objective_signature,
    store_reverse_result,
)

REV_SCRIPT = ROOT / "tests" / "rev_gui_app.py"


@pytest.fixture(autouse=True)
def _contain_shared_module_mutation():
    """Restore any shared module attributes the AppTest harness stubs, so a
    UI-lifecycle test never leaks a fake into later service-level tests."""
    import gui.model_selection as ms

    real_service = svc.reverse_optimize_canonical
    real_selector = ms.render_model_selector
    yield
    svc.reverse_optimize_canonical = real_service
    ms.render_model_selector = real_selector



# ---------------------------------------------------------------------------
# Pure helpers: the deterministic staleness decision
# ---------------------------------------------------------------------------

def _sig(family="FAM", oh=True, t_oh=95.0, nps=False, t_nps=82.0):
    return objective_signature(family, oh, t_oh, nps, t_nps)


def test_signature_ignores_inactive_objective_targets():
    """An unchecked objective contributes None, so a stale disabled-input value
    can never keep an old result alive."""
    assert objective_signature("FAM", True, 95.0, False, 82.0)[2] == 95.0
    assert objective_signature("FAM", True, 95.0, False, 82.0)[4] is None
    assert objective_signature("FAM", False, 99.0, False, 7.0)[2] is None


def test_active_targets_only_include_active_objectives():
    assert active_targets(True, 95.0, False, 82.0) == {"target_oh": 95.0}
    assert active_targets(False, 95.0, True, 82.0) == {"target_nps": 82.0}
    assert active_targets(True, 95.0, True, 82.0) == {
        "target_oh": 95.0, "target_nps": 82.0,
    }
    assert active_targets(False, 95.0, False, 82.0) == {}


def test_unchanged_config_keeps_result_visible():
    session = {}
    sig = _sig()
    store_reverse_result(session, {"ok": True}, sig)
    assert current_reverse_result(session, _sig()) == {"ok": True}
    # Still present after repeated reruns with the SAME config.
    assert current_reverse_result(session, _sig()) == {"ok": True}


def test_changing_oh_target_invalidates_old_result():
    session = {}
    store_reverse_result(session, {"ok": True}, _sig(t_oh=95.0))
    assert current_reverse_result(session, _sig(t_oh=96.0)) is None
    assert REVERSE_RESULT_KEY not in session


def test_changing_nps_target_invalidates_old_result():
    session = {}
    store_reverse_result(
        session, {"ok": True},
        _sig(oh=False, nps=True, t_nps=82.0),
    )
    assert current_reverse_result(
        session, _sig(oh=False, nps=True, t_nps=90.0)
    ) is None
    assert REVERSE_RESULT_KEY not in session


def test_unchecking_oh_invalidates_old_result():
    session = {}
    store_reverse_result(session, {"ok": True}, _sig(oh=True, nps=True))
    assert current_reverse_result(session, _sig(oh=False, nps=True)) is None


def test_unchecking_nps_invalidates_old_result():
    session = {}
    store_reverse_result(session, {"ok": True}, _sig(oh=True, nps=True))
    assert current_reverse_result(session, _sig(oh=True, nps=False)) is None


def test_changing_family_invalidates_old_result():
    session = {}
    store_reverse_result(session, {"ok": True}, _sig(family="FAM"))
    assert current_reverse_result(session, _sig(family="OTHER")) is None


def test_stale_result_never_survives_objective_change():
    """A successful result under one config must not appear once the objective
    selection changes — even if the stale target number is still in state."""
    session = {}
    # Run with OH+NPS, both achieved.
    sig_old = _sig(oh=True, nps=True, t_oh=95.0, t_nps=82.0)
    store_reverse_result(session, {"success": True, "target_oh": 95.0}, sig_old)
    # Now only NPS is active; the stale OH target (95) must not resurrect the old
    # successful OH+NPS result.
    sig_new = _sig(oh=False, nps=True, t_oh=95.0, t_nps=82.0)
    assert current_reverse_result(session, sig_new) is None
    assert REVERSE_RESULT_KEY not in session


# ---------------------------------------------------------------------------
# End-to-end UI lifecycle via Streamlit AppTest
# ---------------------------------------------------------------------------

def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REV_SCRIPT))
    at.run()
    assert not list(at.exception), list(at.exception)
    return at


def _calls(at):
    try:
        return list(at.session_state["_rev_calls"])
    except Exception:
        return []


def _has_result(at):
    return "reverse_result" in at.session_state


def _result_header_rendered(at):
    return any(
        "REVERSE OPTIMIZATION RESULT" in m.value for m in at.markdown
    )


def test_ui_run_oh_only_passes_active_target_only():
    at = _app()
    at.button[0].click()
    at.run()
    assert _calls(at)[-1] == {"family": "FAM", "target_oh": 95.0}
    assert _has_result(at)
    assert _result_header_rendered(at)
    assert any("Target reached" in s.value for s in at.success)


def test_ui_run_nps_only_passes_active_target_only():
    at = _app()
    at.checkbox(key="rev_optimise_oh").set_value(False).run()
    at.checkbox(key="rev_optimise_nps").set_value(True).run()
    at.button[0].click()
    at.run()
    assert _calls(at)[-1] == {"family": "FAM", "target_nps": 82.0}
    assert _has_result(at)


def test_ui_run_joint_passes_both_active_targets():
    at = _app()
    at.checkbox(key="rev_optimise_nps").set_value(True).run()
    at.button[0].click()
    at.run()
    assert _calls(at)[-1] == {"family": "FAM", "target_oh": 95.0, "target_nps": 82.0}
    assert _has_result(at)


def test_ui_changing_oh_target_invalidates_old_result():
    at = _app()
    at.button[0].click()
    at.run()
    assert _has_result(at)
    at.number_input(key="rev_target_oh").set_value(96.0).run()
    assert not _has_result(at)
    assert not _result_header_rendered(at)


def test_ui_changing_nps_target_invalidates_old_result():
    at = _app()
    at.checkbox(key="rev_optimise_nps").set_value(True).run()
    at.button[0].click()
    at.run()
    assert _has_result(at)
    at.number_input(key="rev_target_nps").set_value(90.0).run()
    assert not _has_result(at)
    assert not _result_header_rendered(at)


def test_ui_unchecking_oh_invalidates_old_result():
    at = _app()
    at.checkbox(key="rev_optimise_nps").set_value(True).run()
    at.button[0].click()
    at.run()
    assert _has_result(at)
    at.checkbox(key="rev_optimise_oh").set_value(False).run()
    assert not _has_result(at)
    assert not _result_header_rendered(at)


def test_ui_unchecking_nps_invalidates_old_result():
    at = _app()
    at.checkbox(key="rev_optimise_nps").set_value(True).run()
    at.button[0].click()
    at.run()
    assert _has_result(at)
    at.checkbox(key="rev_optimise_nps").set_value(False).run()
    assert not _has_result(at)
    assert not _result_header_rendered(at)


def test_ui_changing_family_invalidates_old_result():
    at = _app()
    at.button[0].click()
    at.run()
    assert _has_result(at)
    at.session_state["_rev_family"] = "OTHER"
    at.run()
    assert not _has_result(at)
    assert not _result_header_rendered(at)


def test_ui_neither_objective_does_not_call_optimizer():
    at = _app()
    at.button[0].click()
    at.run()
    base = len(_calls(at))
    # Deselect both objectives.
    at.checkbox(key="rev_optimise_oh").set_value(False).run()
    assert any("Select at least one objective" in w.value for w in at.warning)
    assert at.button[0].disabled is True
    at.button[0].click()
    at.run()
    # No new optimizer call, and no stale result shown.
    assert len(_calls(at)) == base
    assert not _has_result(at)
    assert not _result_header_rendered(at)


def test_ui_result_stays_visible_across_unchanged_reruns():
    """An internal rerun that does NOT change the config must keep the result
    visible (requirement 5)."""
    at = _app()
    at.button[0].click()
    at.run()
    assert _has_result(at)
    # Rerun with no interaction -> still visible.
    at.run()
    assert _has_result(at)
    assert _result_header_rendered(at)


# ---------------------------------------------------------------------------
# Service-level: joint candidates keep predicted OH/NPS + canonical MC interval
# ---------------------------------------------------------------------------

_MC = {"p05": 60.0, "p50": 84.0, "p95": 97.0,
       "basis": "monte_carlo_survey_score_distribution"}


def _candidate(rank, oh, nps):
    return {
        "rank": rank,
        "name": f"Candidate {rank}",
        "generated": True,
        "source": "reverse_optimizer_generated_state",
        "state": {"quality": 92.0, "release": 62.0, "transfer": 8.0},
        "state_changes": {"quality": 5.0},
        "predicted_operations_health": oh,
        "target_operations_health": 95.0,
        "operations_health_error": abs(oh - 95.0),
        "predicted_nps": nps,
        "target_nps": 85.0,
        "nps_error": abs(nps - 85.0),
        "feasible": True,
        "confidence_interval": _MC,
        "explanation": "Generated state.",
        "rank_reason": "Meets target.",
        "distance_to_target": 0.05,
        "joint_oh_nps_distance": 0.1,
        "optimization_basis": "joint_operations_health_and_nps",
    }


class _JointEngine:
    def __init__(self, response):
        self._response = response

    def execute(self, request):
        assert request.parameters.get("state") is not None
        return self._response


def _joint_response():
    return ForecastResponse(
        success=True,
        operation="reverse_optimize",
        engine="ReverseOptimizer",
        timestamp=datetime.datetime.now().isoformat(),
        warnings=[],
        errors=[],
        metadata={
            "ranked_candidates": [
                _candidate(1, 95.01, 85.02),
                _candidate(2, 93.5, 82.0),
            ],
            "best_effort": False,
            "target_achieved": True,
            "timed_out": False,
            "no_objective": False,
        },
        payload={
            "success": True,
            "best_solution": {
                "predicted_operations_health": 95.01,
                "predicted_nps": 85.02,
                "state": {"quality": 92.0, "release": 62.0, "transfer": 8.0},
                "state_changes": {"quality": 5.0},
                "distance_to_target": 0.05,
            },
            "metadata": {
                "ranked_candidates": [
                    _candidate(1, 95.01, 85.02),
                    _candidate(2, 93.5, 82.0),
                ],
                "best_effort": False,
                "target_achieved": True,
            },
        },
    )


def test_joint_candidates_keep_predicted_oh_nps_and_mc_interval(monkeypatch):
    monkeypatch.setattr(
        "core.forecast_ai.engines.reverse_optimizer.ReverseOptimizer",
        lambda *a, **k: _JointEngine(_joint_response()),
    )
    STATE.set_active_family("production")
    payload = svc.reverse_optimize_canonical(target_oh=95.0, target_nps=85.0)
    assert payload["success"] is True
    cands = payload["candidates"]
    assert len(cands) == 2
    for cand in cands:
        assert cand["predicted_operations_health"] is not None
        assert cand["predicted_nps"] is not None
        ci = cand.get("confidence_interval") or {}
        assert ci.get("p05") is not None and ci.get("p95") is not None
        assert ci.get("basis") == "monte_carlo_survey_score_distribution"


def test_inactive_objective_is_passed_as_none_at_service_boundary(monkeypatch):
    """When only NPS is active, reverse_optimize_canonical must receive
    target_oh=None (and vice-versa)."""
    seen = {}

    class Recorder(_JointEngine):
        def execute(self, request):
            seen.update(dict(request.parameters))
            return self._response

    monkeypatch.setattr(
        "core.forecast_ai.engines.reverse_optimizer.ReverseOptimizer",
        lambda *a, **k: Recorder(_joint_response()),
    )
    STATE.set_active_family("production")

    svc.reverse_optimize_canonical(target_nps=85.0)
    assert seen.get("target_oh") is None
    assert seen.get("target_nps") == 85.0
