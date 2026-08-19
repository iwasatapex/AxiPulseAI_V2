"""Regression tests for the Streamlit Reverse Optimization UI.

These cover the canonical joint OH+NPS reverse path that the GUI now uses:

  1. Clicking Reverse Optimization with target OH produces a visible result
     payload/renderable result (OH and NPS predicted from the same state).
  2. Clicking with target NPS produces a visible result (predicted OH included).
  3. Clicking with both OH and NPS produces a joint result.
  4. Unreachable targets still render the closest generated state and candidates.
  5. Optimizer exceptions/errors are surfaced instead of silently disappearing.
  6. The GUI does not call TargetStateEngine for the reverse OH/NPS path.
  7. Candidate data survives the GUI/service pipeline.
  8. The result uses the canonical production MC NPS interval.
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


# ---------------------------------------------------------------------------
# Fake canonical ReverseOptimizer engine (service-level tests)
# ---------------------------------------------------------------------------

def _mc_interval(basis="monte_carlo_survey_score_distribution"):
    return {"p05": 60.0, "p50": 84.0, "p95": 97.0, "basis": basis}


def _candidate(rank, oh, nps, feasible=True, name=None, interval=None):
    return {
        "rank": rank,
        "name": name or f"Candidate {rank}",
        "generated": True,
        "source": "reverse_optimizer_generated_state",
        "state": {"quality": 92.0, "release": 62.0, "transfer": 8.0},
        "state_changes": {"quality": 5.0, "release": 2.0, "transfer": -1.0},
        "key_operational_changes": {"quality": 5.0, "release": 2.0},
        "predicted_operations_health": oh,
        "target_operations_health": 95.0,
        "operations_health_error": abs(oh - 95.0) if oh is not None else None,
        "predicted_nps": nps,
        "target_nps": 85.0,
        "nps_error": abs(nps - 85.0) if nps is not None else None,
        "feasible": feasible,
        "confidence_interval": interval if interval is not None else _mc_interval(),
        "explanation": "Generated state evaluated through the canonical service.",
        "objective_score": 0.1,
        "distance_to_target": 0.05,
        "joint_oh_nps_distance": 0.1,
        "rank_reason": "Meets the target OH and NPS within tolerance; ranked by objective score.",
        "optimization_basis": "joint_operations_health_and_nps",
    }


def _best_solution(oh=95.01, nps=85.02):
    return {
        "predicted_operations_health": oh,
        "predicted_nps": nps,
        "state_changes": {"quality": 5.0, "release": 2.0, "transfer": -1.0},
        "applied_scenarios": [],
        "optimization_score": 0.05,
        "distance_to_target": 0.02,
        "iterations_used": 5,
        "constraints_satisfied": True,
        "state": {"quality": 92.0, "release": 62.0, "transfer": 8.0},
        "metadata": {"is_original": False},
    }
def _engine_response(success=True, errors=None, warnings=None,
                     best=None, candidates=None, best_effort=False):
    metadata = {
        "ranked_candidates": candidates or [_candidate(1, 95.01, 85.02)],
        "best_effort": best_effort,
        "target_achieved": success,
        "timed_out": False,
        "no_objective": False,
    }
    return ForecastResponse(
        success=success,
        operation="reverse_optimize",
        engine="ReverseOptimizer",
        timestamp=datetime.datetime.now().isoformat(),
        warnings=list(warnings or []),
        errors=list(errors or []),
        metadata={"phase": "7"},
        payload={
            "success": success,
            "solutions": [],
            "best_solution": best if best is not None else _best_solution(),
            "warnings": list(warnings or []),
            "errors": list(errors or []),
            "metadata": metadata,
        },
    )


class FakeReverseEngine:
    """Injects a deterministic canonical-optimizer payload into the service."""

    def __init__(self, response):
        self._response = response

    def execute(self, request):
        assert request.parameters.get("state") is not None
        return self._response


def _patch_engine(monkeypatch, response):
    engine_cls = FakeReverseEngine(response)
    monkeypatch.setattr(
        "core.forecast_ai.engines.reverse_optimizer.ReverseOptimizer",
        lambda *a, **k: engine_cls,
    )
    STATE.set_active_family("production")


def _assert_renderable(payload):
    """Common assertions that the GUI can render this payload."""
    assert payload["success"] is True
    assert payload["predicted_oh"] is not None
    assert payload["predicted_nps"] is not None
    assert payload["recommended_state"]  # recommended state present
    assert payload["candidates"]  # multiple / at least one candidate exposed
    # Canonical MC interval present on candidates (not scalar noise).
    for c in payload["candidates"]:
        ci = c.get("confidence_interval") or {}
        assert ci.get("p05") is not None and ci.get("p95") is not None
        assert ci.get("basis") == "monte_carlo_survey_score_distribution"

# ---------------------------------------------------------------------------
# 1. Clicking with target OH produces a visible result payload
# ---------------------------------------------------------------------------

def test_reverse_optimize_oh_target_visible_result(monkeypatch):
    _patch_engine(monkeypatch, _engine_response())
    payload = svc.reverse_optimize_canonical(target_oh=95.0)
    _assert_renderable(payload)
    # OH-only target still predicts NPS from the same generated state.
    assert payload["predicted_nps"] is not None
    assert payload["target_oh"] == 95.0
    assert payload["metric"] == "OH+NPS"


# ---------------------------------------------------------------------------
# 2. Clicking with target NPS produces a visible result (predicted OH included)
# ---------------------------------------------------------------------------

def test_reverse_optimize_nps_target_visible_result(monkeypatch):
    _patch_engine(monkeypatch, _engine_response())
    payload = svc.reverse_optimize_canonical(target_nps=85.0)
    _assert_renderable(payload)
    # NPS-only target still predicts OH from the same generated state.
    assert payload["predicted_oh"] is not None
    assert payload["target_nps"] == 85.0


# ---------------------------------------------------------------------------
# 3. Clicking with both OH and NPS produces a joint result
# ---------------------------------------------------------------------------

def test_reverse_optimize_joint_oh_and_nps(monkeypatch):
    _patch_engine(monkeypatch, _engine_response())
    payload = svc.reverse_optimize_canonical(target_oh=95.0, target_nps=85.0)
    _assert_renderable(payload)
    assert payload["target_oh"] == 95.0
    assert payload["target_nps"] == 85.0
    assert payload["predicted_oh"] is not None
    assert payload["predicted_nps"] is not None
    # Joint basis carried on candidates.
    assert all(
        c.get("optimization_basis") == "joint_operations_health_and_nps"
        for c in payload["candidates"]
    )


# ---------------------------------------------------------------------------
# 4. Unreachable targets still render the closest generated state + candidates
# ---------------------------------------------------------------------------

def test_reverse_optimize_unreachable_still_renders(monkeypatch):
    _patch_engine(
        monkeypatch,
        _engine_response(
            success=False,
            best=_best_solution(oh=92.0, nps=80.0),
            candidates=[
                _candidate(1, 92.0, 80.0, feasible=False, name="Candidate 1"),
                _candidate(2, 91.5, 79.5, feasible=False, name="Candidate 2"),
            ],
        ),
    )
    payload = svc.reverse_optimize_canonical(target_oh=99.0)
    assert payload["success"] is False
    assert payload["found"] is True  # closest generated state present
    assert payload["predicted_oh"] == 92.0
    assert payload["predicted_nps"] == 80.0
    assert len(payload["candidates"]) == 2
    assert "not reachable" in payload["status"]
    # Each candidate remains renderable with both OH and NPS.
    for c in payload["candidates"]:
        assert c["predicted_operations_health"] is not None
        assert c["predicted_nps"] is not None

# ---------------------------------------------------------------------------
# 5. Optimizer exceptions/errors are surfaced, never silently swallowed
# ---------------------------------------------------------------------------

def test_reverse_optimize_errors_are_surfaced(monkeypatch):
    _patch_engine(
        monkeypatch,
        _engine_response(
            success=False,
            errors=["Optimization error: model exploded"],
            best=None,
            candidates=[],
        ),
    )
    payload = svc.reverse_optimize_canonical(target_oh=95.0)
    assert payload["success"] is False
    assert payload["errors"]
    assert "Optimization error: model exploded" in payload["errors"]


def test_reverse_optimize_abstains_without_target(monkeypatch):
    _patch_engine(monkeypatch, _engine_response())
    payload = svc.reverse_optimize_canonical()
    assert payload["success"] is False
    assert payload["abstained"] is True
    assert payload["errors"]
    assert payload["candidates"] == []
    # Never silently invents an objective.
    assert "objective" in payload["errors"][0].lower()


# ---------------------------------------------------------------------------
# 6. The GUI does not call TargetStateEngine for the reverse OH/NPS path
# ---------------------------------------------------------------------------

def test_reverse_optimize_gui_path_never_calls_target_state_engine(monkeypatch):
    # The canonical reverse path must not reach find_target_state (which wraps
    # TargetStateEngine). Make it fail loudly if it is ever invoked.
    def _boom(*a, **k):
        raise AssertionError("TargetStateEngine/find_target_state was called")

    monkeypatch.setattr(svc, "find_target_state", _boom)
    _patch_engine(monkeypatch, _engine_response())
    payload = svc.reverse_optimize_canonical(target_oh=95.0)
    _assert_renderable(payload)


# ---------------------------------------------------------------------------
# 7. Candidate data survives the GUI/service pipeline
# ---------------------------------------------------------------------------

def test_reverse_optimize_candidates_survive_service_pipeline(monkeypatch):
    candidates = [
        _candidate(1, 95.01, 85.02, feasible=True),
        _candidate(2, 93.5, 82.0, feasible=False, name="Candidate 2"),
        _candidate(3, 92.0, 80.0, feasible=False, name="Candidate 3"),
    ]
    _patch_engine(monkeypatch, _engine_response(candidates=candidates))
    payload = svc.reverse_optimize_canonical(target_oh=95.0)
    assert len(payload["candidates"]) == 3
    assert [c["rank"] for c in payload["candidates"]] == [1, 2, 3]
    # Each candidate's OH+NPS + explanation + rank reason survive intact.
    c0 = payload["candidates"][0]
    assert c0["predicted_operations_health"] == 95.01
    assert c0["predicted_nps"] == 85.02
    assert c0["feasible"] is True
    assert c0["explanation"]
    assert c0["rank_reason"]
    assert c0["state"]  # recommended state on the candidate
    # The view renders candidates up to MAX_EXPOSED_CANDIDATES.
    from gui.views.reverse_view import MAX_EXPOSED_CANDIDATES
    assert MAX_EXPOSED_CANDIDATES == 7


# ---------------------------------------------------------------------------
# 8. The result uses the canonical production MC NPS interval
# ---------------------------------------------------------------------------

def test_reverse_optimize_uses_canonical_mc_nps_interval(monkeypatch):
    _patch_engine(monkeypatch, _engine_response())
    payload = svc.reverse_optimize_canonical(target_oh=95.0)
    c0 = payload["candidates"][0]
    ci = c0["confidence_interval"]
    assert ci["basis"] == "monte_carlo_survey_score_distribution"
    assert ci["p05"] < ci["p50"] < ci["p95"]
    # The GUI exposes the canonical 90% interval on the candidate.
    assert ci["p05"] == 60.0 and ci["p95"] == 97.0
