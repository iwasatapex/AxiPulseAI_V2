"""
Search strategies for Reverse Optimizer.
This module contains a deterministic hill climber that searches for
operational states that achieve a target prediction.
"""
import math
import time
from copy import deepcopy
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass

from ..state import OperationalState
from ..config import KPI_BOUNDS
from .models import TargetGoal, Constraint, OptimizationSolution
from .constraints import ConstraintValidator
from .scoring import ScoreCalculator

class DeterministicHillClimb:
    """
    Deterministic hill climbing that explores a fixed set of step sizes
    and directions. No randomness; repeatable.
    """
    def __init__(self, step_sizes: List[float] = None, fields: List[str] = None):
        self.step_sizes = step_sizes or [0.5, 1.0, 2.0]
        self.fields = fields or ['quality', 'competency', 'attendance', 'release', 'transfer']
        self.bounds = KPI_BOUNDS
        self.timed_out = False

    def _clamp(self, state: OperationalState) -> OperationalState:
        """Clamp values to config-defined bounds."""
        new_state = deepcopy(state)
        for field in self.fields:
            if field in self.bounds:
                min_val, max_val = self.bounds[field]
                current = getattr(new_state, field, 0.0)
                setattr(new_state, field, max(min_val, min(max_val, current)))
        return new_state

    def _generate_candidates(self, current: OperationalState) -> List[OperationalState]:
        """Generate candidate states deterministically.

        In addition to the small neighborhood steps, a set of boundary
        candidates is produced (each field at its clamp bound, plus the
        all-max / all-min states). This lets the search reliably reach the
        attainable target range instead of stalling in a local optimum near
        the boundary (P1 reliability fix). All candidates stay within config
        bounds; user constraints are validated separately by the caller.
        """
        candidates: List[OperationalState] = []

        def _mk(overrides) -> OperationalState:
            candidate = deepcopy(current)
            for field, value in overrides.items():
                setattr(candidate, field, value)
            return self._clamp(candidate)

        # Boundary / extreme candidates (deterministic, respects bounds).
        # Single-field extremes are tried first because, for realistic
        # improvement targets, the attainable optimum is usually reached by
        # pushing one or a few KPIs to their bound (verified empirically). This
        # keeps the search within the timeout budget instead of wasting evals.
        for field in self.fields:
            lo, hi = self.bounds.get(field, (0, 100))
            candidates.append(_mk({field: hi}))
            candidates.append(_mk({field: lo}))
        max_state = _mk({f: self.bounds.get(f, (0, 100))[1] for f in self.fields})
        min_state = _mk({f: self.bounds.get(f, (0, 100))[0] for f in self.fields})
        candidates.append(max_state)
        candidates.append(min_state)

        # Neighborhood steps (existing behavior).
        for field in self.fields:
            for step in self.step_sizes:
                for direction in [-1, 1]:
                    candidate = deepcopy(current)
                    current_val = getattr(candidate, field, 0.0)
                    new_val = current_val + direction * step
                    setattr(candidate, field, new_val)
                    candidate = self._clamp(candidate)
                    candidates.append(candidate)
        return candidates

    def iterate(self, current_state: OperationalState,
                evaluator: Callable[[OperationalState], Tuple[Optional[float], Optional[float]]],
                target_goal: TargetGoal,
                constraints: List[Constraint],
                max_iterations: int,
                timeout_seconds: int) -> List[OptimizationSolution]:
        start_time = time.time()
        # Genuine timeout enforcement (Phase 5): the deadline is checked INSIDE
        # the candidate loop (before every model evaluation), not only between
        # iterations, so a slow evaluator cannot run past the budget.
        deadline = start_time + max(0.0, float(timeout_seconds))
        self.timed_out = False
        original_state = deepcopy(current_state)
        best_state = current_state
        best_oh = None
        best_nps = None
        best_distance = float('inf')
        solutions = []
        iteration = 0

        # Evaluate the original (do-nothing) state first: a target already met
        # by the current state succeeds immediately and deterministically,
        # without spending the optimization budget. The evaluated state is a
        # deepcopy, so the caller's state is never mutated.
        original_oh, original_nps = evaluator(original_state)
        if original_oh is not None and original_nps is not None:
            original_distance = ScoreCalculator.compute_distance(
                original_state, original_oh, original_nps, target_goal
            )
            zero_changes = {f: 0.0 for f in self.fields}
            solutions.append(OptimizationSolution(
                predicted_operations_health=original_oh,
                predicted_nps=original_nps,
                state_changes=zero_changes,
                applied_scenarios=[],
                optimization_score=ScoreCalculator.compute_score(
                    OptimizationSolution(
                        original_oh, original_nps, zero_changes, [], 0.0,
                        original_distance, 0, True, {},
                    ),
                    original_state,
                    target_goal,
                ),
                distance_to_target=original_distance,
                iterations_used=0,
                constraints_satisfied=True,
                state={f: getattr(original_state, f, 0.0) for f in self.fields},
                metadata={"is_original": True},
            ))
            best_oh = original_oh
            best_nps = original_nps
            best_distance = original_distance
            best_state = original_state
            if ScoreCalculator.is_acceptable(original_distance, target_goal):
                # Target already met by the current state.
                solutions.sort(key=lambda s: s.optimization_score)
                return solutions

        found_acceptable = False
        while iteration < max_iterations:
            if time.time() > deadline:
                self.timed_out = True
                break

            # Generate candidates from best_state
            candidates = self._generate_candidates(best_state)
            improved = False

            for candidate in candidates:
                # Enforce the deadline before each expensive evaluation.
                if time.time() > deadline:
                    self.timed_out = True
                    break
                # Validate constraints
                if not ConstraintValidator.validate(candidate, constraints):
                    continue
                if not ConstraintValidator.validate_change(original_state, candidate, constraints):
                    continue

                # Evaluate
                oh, nps = evaluator(candidate)
                if oh is None or nps is None:
                    continue

                distance = ScoreCalculator.compute_distance(candidate, oh, nps, target_goal)

                # Build solution object (for storage)
                changes = {f: getattr(candidate, f, 0.0) - getattr(original_state, f, 0.0)
                           for f in self.fields}
                solution = OptimizationSolution(
                    predicted_operations_health=oh,
                    predicted_nps=nps,
                    state_changes=changes,
                    applied_scenarios=[],
                    optimization_score=ScoreCalculator.compute_score(
                        OptimizationSolution(oh, nps, changes, [], 0.0, distance, iteration, True, {}),
                        original_state,
                        target_goal
                    ),
                    distance_to_target=distance,
                    iterations_used=iteration,
                    constraints_satisfied=True,
                    state={f: getattr(candidate, f, 0.0) for f in self.fields}
                )
                solutions.append(solution)

                # Update best if improvement
                if distance < best_distance:
                    best_distance = distance
                    best_state = candidate
                    best_oh = oh
                    best_nps = nps
                    improved = True
                    # Target reached: stop the entire search immediately,
                    # not just the inner candidate scan. Without this the
                    # outer loop keeps iterating until the deadline even
                    # though an acceptable solution was already found (P1).
                    if ScoreCalculator.is_acceptable(distance, target_goal):
                        found_acceptable = True
                        break

            if found_acceptable or self.timed_out or not improved:
                # Acceptable solution found, budget exhausted, or no better
                # candidate; stop.
                break

            iteration += 1

        # Rank solutions by score
        solutions.sort(key=lambda s: s.optimization_score)
        return solutions

iterate = DeterministicHillClimb.iterate
