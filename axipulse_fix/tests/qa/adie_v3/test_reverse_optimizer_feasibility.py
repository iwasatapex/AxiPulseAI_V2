"""ReverseOptimizer / DeterministicHillClimb feasibility regression (P1).

Covers the five required cases using a deterministic mock evaluator so the
test is fast and model-independent:

  1. target already met  -> success (do-nothing, no wasted budget)
  2. realistic feasible improvement -> success
  3. genuinely infeasible target -> explicit failure (no fabricated success)
  4. timeout -> explicit timeout
  5. repeated identical optimization -> identical result
"""

from core.forecast_ai.optimization.search import DeterministicHillClimb
from core.forecast_ai.optimization.models import TargetGoal
from core.forecast_ai.state import OperationalState


def _evaluator_factory():
    """OH = 80 + 0.1*quality (deterministic, monotonic), NPS = 70 + 0.05*competency."""
    def evaluate(state: OperationalState):
        oh = 80.0 + state.quality * 0.1
        nps = 70.0 + state.competency * 0.05
        return oh, nps
    return evaluate


def _state():
    return OperationalState(
        quality=80.0, competency=70.0, transfer=10.0,
        release=85.0, attendance=90.0, operations_health=88.0,
    )


def _run(target_oh, timeout=10, tol=1.0):
    evaluator = _evaluator_factory()
    searcher = DeterministicHillClimb()
    solutions = searcher.iterate(
        _state(),
        evaluator,
        TargetGoal(target_operations_health=target_oh, tolerance=tol),
        [],
        max_iterations=25,
        timeout_seconds=timeout,
    )
    acceptable = [s for s in solutions if s.distance_to_target <= tol]
    return searcher, solutions, acceptable


def test_case1_target_already_met_succeeds():
    # Original OH = 80 + 0.1*80 = 88; target 87 within tolerance 1.0 -> met.
    searcher, solutions, acceptable = _run(87.0)
    assert acceptable, "already-met target must be acceptable"
    assert searcher.timed_out is False, "must not spend budget on already-met target"


def test_case2_feasible_improvement_succeeds():
    # Target 90 is reachable (quality -> 100 gives OH 90). Must succeed and
    # must not require hitting the timeout.
    searcher, solutions, acceptable = _run(90.0)
    assert acceptable, "feasible improvement target must succeed"
    assert searcher.timed_out is False
    best = acceptable[0]
    assert best.predicted_operations_health >= 90.0


def test_case3_infeasible_target_fails_explicitly():
    # Max attainable OH is 90 (quality=100); target 95 is infeasible.
    searcher, solutions, acceptable = _run(95.0, timeout=2)
    assert not acceptable, "infeasible target must not fabricate success"
    # It either ran out of budget (timeout) or could not improve (no solution).
    assert searcher.timed_out or not any(
        s.distance_to_target <= 1.0 for s in solutions
    )


def test_case4_timeout_is_explicit():
    searcher, solutions, acceptable = _run(95.0, timeout=0.001)
    assert searcher.timed_out is True


def test_case5_repeated_identical():
    _, a_sol, a_acc = _run(90.0)
    _, b_sol, b_acc = _run(90.0)
    assert a_acc and b_acc
    assert a_acc[0].state == b_acc[0].state
    assert a_acc[0].predicted_operations_health == b_acc[0].predicted_operations_health
