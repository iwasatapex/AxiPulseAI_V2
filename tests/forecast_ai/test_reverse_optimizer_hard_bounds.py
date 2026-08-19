"""Regression tests proving the reverse optimizer can NEVER generate, select,
or expose an operationally-invalid operational state.

The canonical KPI hard bounds are:
    quality     60..100
    competency  55..100
    attendance  65..100
    release     50..100
    transfer     0..20

A state outside these bounds (e.g. attendance = 0, change -90) is invalid and
must never be returned as a recommended state, exposed as a feasible
candidate, selected as best_solution merely because its OH/NPS distance is
excellent, or used to claim the target was achieved.
"""
from __future__ import annotations

from core.forecast_ai.optimization import (
    ReverseOptimizer,
    OptimizationRequest,
    TargetGoal,
    ConstraintValidator,
)
from core.forecast_ai.optimization.search import DeterministicHillClimb
from core.forecast_ai.prediction.service import PredictionService
from core.forecast_ai.state import OperationalState

# Canonical hard bounds (mirror core/forecast_ai/config.KPI_BOUNDS).
BOUNDS = {
    "quality": (60.0, 100.0),
    "competency": (55.0, 100.0),
    "attendance": (65.0, 100.0),
    "release": (50.0, 100.0),
    "transfer": (0.0, 20.0),
}

BASE = {"quality": 87.0, "competency": 93.0, "attendance": 90.0,
        "release": 60.0, "transfer": 9.0}


def _in_bounds(state) -> bool:
    d = state.to_dict() if hasattr(state, "to_dict") else dict(state)
    return all(
        lo <= float(d[f]) <= hi for f, (lo, hi) in BOUNDS.items()
    )


# ---------------------------------------------------------------------------
# Direct canonical hard-bounds validation per KPI
# ---------------------------------------------------------------------------

def test_validate_hard_bounds_attendance_never_below_65():
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "attendance": 64.9})
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "attendance": 0.0})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "attendance": 65.0})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "attendance": 100.0})


def test_validate_hard_bounds_quality_never_below_60_or_above_100():
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "quality": 59.9})
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "quality": 100.1})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "quality": 60.0})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "quality": 100.0})


def test_validate_hard_bounds_competency_never_below_55_or_above_100():
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "competency": 54.9})
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "competency": 100.1})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "competency": 55.0})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "competency": 100.0})


def test_validate_hard_bounds_release_never_below_50_or_above_100():
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "release": 49.9})
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "release": 100.1})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "release": 50.0})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "release": 100.0})


def test_validate_hard_bounds_transfer_never_below_0_or_above_20():
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "transfer": -0.1})
    assert not ConstraintValidator.validate_hard_bounds({**BASE, "transfer": 20.1})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "transfer": 0.0})
    assert ConstraintValidator.validate_hard_bounds({**BASE, "transfer": 20.0})


# ---------------------------------------------------------------------------
# Candidate generation itself stays within the canonical bounds
# ---------------------------------------------------------------------------

def test_candidate_generation_never_emits_out_of_bounds_state():
    searcher = DeterministicHillClimb()
    state = OperationalState.from_dict(dict(BASE))
    for _ in range(200):
        for cand in searcher._generate_candidates(state):
            assert _in_bounds(cand), f"generated invalid candidate: {cand.to_dict()}"
        state = searcher._clamp(searcher._generate_candidates(state)[0])


# ---------------------------------------------------------------------------
# Predictors that REWARD out-of-bounds states (strongest possible temptation)
# ---------------------------------------------------------------------------

class TemptLowAttendanceOH:
    """OH rises as attendance falls — an invalid (low) attendance looks perfect."""

    evaluated: list = []

    def predict(self, state):
        a = float(state.get("attendance", 0.0))
        q = float(state.get("quality", 0.0))
        c = float(state.get("competency", 0.0))
        r = float(state.get("release", 0.0))
        TemptLowAttendanceOH.evaluated.append(dict(state))
        return 100.0 - a + 0.3 * q + 0.2 * c + 0.05 * r


class TemptLowAttendanceNPS:
    """NPS rises as attendance falls — an invalid (low) attendance looks perfect."""

    evaluated: list = []

    def predict(self, state):
        a = float(state.get("attendance", 0.0))
        q = float(state.get("quality", 0.0))
        c = float(state.get("competency", 0.0))
        r = float(state.get("release", 0.0))
        TemptLowAttendanceNPS.evaluated.append(dict(state))
        return {"nps": 60.0 - a + 0.3 * q + 0.2 * c + 0.05 * r}


def _tempting_opt():
    TemptLowAttendanceOH.evaluated = []
    TemptLowAttendanceNPS.evaluated = []
    service = PredictionService(
        oh_predictor=TemptLowAttendanceOH(),
        nps_predictor=TemptLowAttendanceNPS(),
    )
    return ReverseOptimizer(prediction_service=service)


def _run(opt, target, mi=200, to=60):
    return opt.optimize(OptimizationRequest(
        initial_state=dict(BASE), target_goal=target,
        max_iterations=mi, timeout_seconds=to,
    ))


def test_invalid_low_attendance_never_evaluated_or_selected():
    """Even though attendance=0 yields a PERFECT OH/NPS, the optimizer must
    never evaluate it (rejected at generation) and best_solution must remain a
    valid operational state."""
    opt = _tempting_opt()
    # Target 95 OH / 85 NPS is reachable ONLY via invalid (low) attendance.
    res = _run(opt, TargetGoal(
        target_operations_health=95.0, target_nps=85.0, tolerance=0.5,
    ))

    # The invalid state was never even evaluated by the forward model.
    seen = [float(s.get("attendance", 0.0))
            for s in TemptLowAttendanceOH.evaluated]
    assert seen, "optimizer must evaluate candidate states"
    assert min(seen) >= 65.0, "attendance below 65 must never be evaluated"

    # best_solution is always a valid operational state.
    assert res.best_solution is not None
    assert _in_bounds(res.best_solution.state)
    assert float(res.best_solution.state["attendance"]) >= 65.0


def test_oh_nps_perfect_invalid_state_rejected_and_not_claimed_achieved():
    """An OH/NPS-perfect but operationally-invalid state must never be
    reported as achieved or exposed as a feasible candidate."""
    opt = _tempting_opt()
    res = _run(opt, TargetGoal(
        target_operations_health=95.0, target_nps=85.0, tolerance=0.5,
    ))

    # Valid states cannot reach the target (best valid OH=90, NPS=50), so this
    # must NOT report success merely because an invalid state would be perfect.
    assert res.success is False
    assert res.metadata.get("target_achieved") is False

    # No feasible candidate may carry an invalid state.
    for c in res.metadata.get("ranked_candidates") or []:
        assert _in_bounds(c["state"]), f"invalid exposed candidate: {c['state']}"
        if c["feasible"]:
            assert _in_bounds(c["state"])


def test_best_solution_always_valid_operational_state():
    """Across several targets (including reachable and unreachable), the
    selected best_solution state must always respect the hard bounds."""
    opt = _tempting_opt()
    targets = [
        TargetGoal(target_operations_health=70.0, tolerance=0.5),  # easy
        TargetGoal(target_operations_health=95.0, tolerance=0.5),  # needs bounds
        TargetGoal(target_operations_health=90.0, target_nps=60.0, tolerance=0.5),
        TargetGoal(target_operations_health=120.0, tolerance=0.5),  # unreachable
    ]
    for target in targets:
        res = _run(opt, target)
        if res.best_solution is not None:
            assert _in_bounds(res.best_solution.state), \
                f"best_solution invalid for {target}: {res.best_solution.state}"


def test_every_exposed_candidate_has_valid_state():
    """Every exposed candidate summary (the raw JSON the GUI reads) must carry a
    valid operational state."""
    opt = _tempting_opt()
    # Infeasible target -> many candidates exposed.
    res = _run(opt, TargetGoal(target_operations_health=120.0, tolerance=0.5))
    cands = res.metadata.get("ranked_candidates") or []
    assert len(cands) >= 1
    for c in cands:
        assert _in_bounds(c["state"]), f"invalid candidate state: {c['state']}"
        # state_changes derived from valid initial -> valid generated state.
        for field in BOUNDS:
            assert field in c["state"]
            assert field in c["state_changes"]


# ---------------------------------------------------------------------------
# GUI recommended_state + raw JSON carry only valid values
# ---------------------------------------------------------------------------

def test_gui_recommended_state_and_candidates_contain_only_valid_values(monkeypatch):
    """End-to-end through the GUI service with the REAL core optimizer (mock
    predictors): the GUI recommended_state and every candidate carry only valid
    operational values."""
    from gui import services as svc
    from gui.state import STATE

    from core.forecast_ai.engines.reverse_optimizer import (
        ReverseOptimizer as Engine,
    )
    from core.forecast_ai.optimization import ReverseOptimizer as CoreOptimizer

    TemptLowAttendanceOH.evaluated = []
    TemptLowAttendanceNPS.evaluated = []
    service = PredictionService(
        oh_predictor=TemptLowAttendanceOH(),
        nps_predictor=TemptLowAttendanceNPS(),
    )
    core = CoreOptimizer(prediction_service=service)
    engine = Engine(optimizer_core=core)

    monkeypatch.setattr(
        "core.forecast_ai.engines.reverse_optimizer.ReverseOptimizer",
        lambda *a, **k: engine,
    )
    STATE.set_active_family("production")

    payload = svc.reverse_optimize_canonical(
        target_oh=95.0, target_nps=85.0, tolerance=0.5,
    )

    recommended = payload["recommended_state"]
    assert recommended, "recommended_state must be present"
    assert _in_bounds(recommended), f"invalid recommended_state: {recommended}"

    for c in payload["candidates"]:
        assert _in_bounds(c["state"]), f"invalid candidate state: {c['state']}"
        if c["feasible"]:
            assert _in_bounds(c["state"])


def test_state_changes_derived_from_valid_initial_to_valid_generated():
    """state_changes are computed between the (clamped) valid initial state and
    the valid generated state, so an out-of-bounds value can never appear."""
    opt = _tempting_opt()
    res = _run(opt, TargetGoal(
        target_operations_health=95.0, target_nps=85.0, tolerance=0.5,
    ))
    if res.best_solution is not None:
        changes = res.best_solution.state_changes
        for field, (lo, hi) in BOUNDS.items():
            assert field in changes
            # Initial (clamped) and generated state are both in-bounds.
            assert lo <= float(BASE[field]) + changes[field] <= hi
