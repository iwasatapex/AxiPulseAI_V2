"""
ReverseOptimizer – Orchestrates search using PredictionService and ScenarioManager.
"""
import time
import logging
from typing import Dict, Any, Optional, List
from copy import deepcopy

from .models import OptimizationRequest, OptimizationResult, OptimizationSolution
from .search import DeterministicHillClimb
from .constraints import ConstraintValidator
from .scoring import ScoreCalculator
from ..prediction import PredictionService
from ..models import PredictionRequest
from ..state import OperationalState
from ..scenarios import ScenarioManager

logger = logging.getLogger(__name__)

# How many of the ranked solutions get exposed to the GUI/recommendation
# layer as distinct candidates, and get the (more expensive) canonical
# Bayesian + Monte Carlo NPS interval attached. The search itself may
# evaluate many more states than this; this only bounds how many of the
# already-ranked results are summarized/enriched for display.
MAX_EXPOSED_CANDIDATES = 7


def _joint_target_distance(operations_health: Optional[float], nps: Optional[float], target) -> Optional[float]:
    """Return a normalized joint OH+NPS distance when both targets exist.

    Reverse optimization is a joint state-generation problem: OH and NPS are
    evaluated from the same generated operational state.  The optimizer must
    never optimize one scalar and retrofit the other afterward.
    """
    if operations_health is None and nps is None:
        return None
    parts = []
    if target.target_operations_health is not None and operations_health is not None:
        parts.append(abs(float(operations_health) - float(target.target_operations_health)))
    if target.target_nps is not None and nps is not None:
        parts.append(abs(float(nps) - float(target.target_nps)))
    if not parts:
        return None
    return float(sum(parts))


class ReverseOptimizer:
    def __init__(self,
                 prediction_service: Optional[PredictionService] = None,
                 scenario_manager: Optional[ScenarioManager] = None):
        self.service = prediction_service or PredictionService()
        self.scenario_manager = scenario_manager or ScenarioManager()

    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        start_time = time.time()
        target = request.target_goal

        # Convert initial state to OperationalState
        initial_state = OperationalState.from_dict(request.initial_state)

        # No objective must never invent one: without target_oh / target_nps
        # there is nothing to optimize, so the optimizer refuses to fabricate
        # a target and preserves no-action semantics (success=False).
        if (
            target.target_operations_health is None
            and target.target_nps is None
        ):
            return OptimizationResult(
                success=False,
                solutions=[],
                best_solution=None,
                warnings=[
                    "No optimization objective (target_oh/target_nps) specified."
                ],
                errors=[
                    "No optimization objective specified; refusing to invent a target."
                ],
                metadata={
                    "duration_seconds": time.time() - start_time,
                    "timed_out": False,
                    "timeout_seconds": request.timeout_seconds,
                    "target_achieved": False,
                    "best_effort": False,
                    "ranked_candidates": [],
                    "no_objective": True,
                },
            )

        # Side channel: keyed by the same (field-value tuple) the search uses
        # to dedupe states, this preserves the full PredictionResult (incl.
        # the 0..10 bayesian_score_distribution / score_counts) for whichever
        # candidates the search ends up ranking, WITHOUT changing the
        # evaluator's (oh, nps) contract that DeterministicHillClimb.iterate
        # and its direct unit tests rely on.
        prediction_cache: Dict[tuple, Any] = {}
        fields = ['quality', 'competency', 'attendance', 'release', 'transfer']

        # Setup evaluator: apply scenarios, then call PredictionService
        def evaluator(state: OperationalState) -> tuple:
            # Apply scenarios (if any are active for day 1; we assume day 1 for simplicity)
            # In future, we could pass a day parameter.
            modified_state = self.scenario_manager.apply_scenarios(state, day=1)
            # Predict
            pred_req = PredictionRequest(state=modified_state.to_dict(), metadata={"optimizer": True})
            result = self.service.predict(pred_req)
            key = tuple(getattr(state, f, 0.0) for f in fields)
            prediction_cache[key] = result
            if result.errors:
                return None, None
            return result.operations_health, result.nps

        # Search
        searcher = DeterministicHillClimb()
        solutions = searcher.iterate(
            initial_state,
            evaluator,
            target,
            target.constraints,
            max_iterations=request.max_iterations,
            timeout_seconds=request.timeout_seconds
        )
        timed_out = getattr(searcher, "timed_out", False)

        # Candidate feasibility requires BOTH the OH/NPS target objective
        # satisfied within tolerance AND every operational state variable
        # respecting its canonical hard bound. An operationally-invalid state
        # (e.g. attendance < 65) is never acceptable merely because its
        # OH/NPS distance is excellent.
        def _in_bounds(sol) -> bool:
            return ConstraintValidator.validate_hard_bounds(sol.state)

        valid_solutions = [s for s in solutions if _in_bounds(s)]
        acceptable = [
            s for s in valid_solutions
            if s.distance_to_target <= target.tolerance
        ]
        original_solution = next(
            (s for s in solutions if (s.metadata or {}).get("is_original")),
            None,
        )
        if acceptable:
            best_solution = acceptable[0]  # already sorted (best score first)
            success = True
            best_effort = False
        else:
            best_solution = valid_solutions[0] if valid_solutions else None
            success = False
            # Genuine best-effort: a valid candidate strictly improves toward
            # the target versus the do-nothing (original) state. The target was
            # not reached, so this must never be reported as achieved.
            best_effort = bool(
                best_solution is not None
                and original_solution is not None
                and best_solution.distance_to_target
                < original_solution.distance_to_target - 1e-9
            )

        duration = time.time() - start_time

        ranked_candidates = self._build_candidate_summaries(
            solutions, target, initial_state, prediction_cache, fields
        )

        if timed_out:
            errors = [
                "optimization_timeout: search exceeded the configured "
                "timeout budget and was aborted"
            ]
            # Preserve a genuine improving candidate discovered before the
            # timeout. The target is still NOT achieved, so success remains
            # False. RecommendationEngine may use best_effort=True to produce
            # an explicitly advisory recommendation rather than dropping all
            # useful optimizer evidence.
            success = False
        elif not success:
            errors = ["No solution within tolerance found"]
        else:
            errors = []

        return OptimizationResult(
            success=success,
            solutions=solutions,
            best_solution=best_solution,
            warnings=[],
            errors=errors,
            metadata={
                "duration_seconds": duration,
                "timed_out": timed_out,
                "timeout_seconds": request.timeout_seconds,
                "target_achieved": success,
                "best_effort": best_effort,
                "target_operations_health": target.target_operations_health,
                "target_nps": target.target_nps,
                "tolerance": target.tolerance,
                # Multiple generated-and-evaluated candidates, ranked, each
                # explained and (for the top few) given a canonical
                # Bayesian/Monte Carlo NPS interval. This is what the GUI /
                # RecommendationEngine should render instead of collapsing to
                # best_solution alone.
                "ranked_candidates": ranked_candidates,
            }
        )

    def _build_candidate_summaries(self, solutions, target, initial_state,
                                    prediction_cache, fields) -> List[Dict[str, Any]]:
        """Turn the ranked OptimizationSolution list into GUI-ready candidate
        summaries: rank, predicted/target OH+NPS and their errors, feasibility,
        a probabilistic NPS interval (top candidates only), the key changes,
        a plain-language explanation, the objective score, and why it ranked
        where it did. Candidates are genuinely distinct generated/evaluated
        states — this does not select from a catalog of trained models.
        """
        summaries: List[Dict[str, Any]] = []
        target_oh = target.target_operations_health
        target_nps = target.target_nps

        # `solutions` is sorted by the raw optimization_score, which mixes
        # distance-to-target with a change-magnitude penalty -- a solution
        # that exactly hits the target via a large change can score worse
        # than one that misses the target via a small change. For candidate
        # exposure, feasibility must come first: every feasible solution
        # ranks ahead of every infeasible one, and only ties within each
        # group are broken by score. Without this, an actually-achieving
        # candidate could be omitted from the exposed list entirely.
        def _display_key(sol):
            joint = _joint_target_distance(
                sol.predicted_operations_health,
                sol.predicted_nps,
                target,
            )
            # Feasible first; for a joint OH+NPS target, prefer the smallest
            # combined absolute target error, then the optimizer score.
            if joint is not None:
                return (
                    sol.distance_to_target > target.tolerance,
                    joint,
                    sol.optimization_score,
                )
            return (sol.distance_to_target > target.tolerance, sol.optimization_score)

        display_order = sorted(solutions, key=_display_key)

        for rank, sol in enumerate(display_order[:MAX_EXPOSED_CANDIDATES], start=1):
            is_original = bool((sol.metadata or {}).get("is_original"))
            # Feasibility = target reached within tolerance AND a valid
            # operational state (every variable within its canonical hard
            # bound). A state that is outside the hard bounds must never be
            # flagged feasible even if its OH/NPS distance is perfect.
            feasible = (
                sol.distance_to_target <= target.tolerance
                and ConstraintValidator.validate_hard_bounds(sol.state)
            )

            oh_error = (
                None if target_oh is None or sol.predicted_operations_health is None
                else sol.predicted_operations_health - target_oh
            )
            nps_error = (
                None if target_nps is None or sol.predicted_nps is None
                else sol.predicted_nps - target_nps
            )

            key = tuple(sol.state.get(f, 0.0) for f in fields)
            pred_result = prediction_cache.get(key)
            probabilistic = self._probabilistic_interval(pred_result, sol.state)

            key_changes = {
                f: round(v, 3) for f, v in sol.state_changes.items() if abs(v) >= 0.01
            }

            if is_original:
                explanation = "Current (do-nothing) operational state."
            elif not key_changes:
                explanation = "No material operational change from current state."
            else:
                change_desc = ", ".join(
                    f"{f} {'+' if v > 0 else ''}{v:g}" for f, v in key_changes.items()
                )
                explanation = f"Adjust {change_desc} relative to the current state."

            if feasible:
                rank_reason = "Meets the target OH and NPS within tolerance; ranked by objective score."
            else:
                rank_reason = (
                    "Closest generated candidate to the target; does NOT satisfy the "
                    "requested tolerance and must not be reported as achieved."
                )

            joint_distance = _joint_target_distance(
                sol.predicted_operations_health,
                sol.predicted_nps,
                target,
            )
            summaries.append({
                "rank": rank,
                "name": "Current state" if is_original else f"Candidate {rank}",
                "generated": True,
                "source": "reverse_optimizer_generated_state",
                "state": dict(sol.state),
                "state_changes": dict(sol.state_changes),
                "predicted_operations_health": sol.predicted_operations_health,
                "target_operations_health": target_oh,
                "operations_health_error": oh_error,
                "predicted_nps": sol.predicted_nps,
                "target_nps": target_nps,
                "nps_error": nps_error,
                "feasible": feasible,
                "confidence_interval": probabilistic,
                "key_operational_changes": key_changes,
                "explanation": explanation,
                "objective_score": sol.optimization_score,
                "distance_to_target": sol.distance_to_target,
                "joint_oh_nps_distance": joint_distance,
                "rank_reason": rank_reason,
                "optimization_basis": "joint_operations_health_and_nps",
            })
        return summaries

    def _probabilistic_interval(self, pred_result, state) -> Optional[Dict[str, float]]:
        """Canonical Bayesian + Monte Carlo NPS interval for a candidate.

        This reads the interval that the PRODUCTION PredictionService already
        computed and preserved on the result (monte_carlo_nps_p05/p50/p95,
        derived from the 0..10 survey-score distribution by the canonical
        Bayesian/Monte-Carlo path). There is deliberately NO independent
        reverse-optimizer re-derivation of the survey/distribution/interval
        here: the candidate interval is exactly the production NPS interval.
        """
        if pred_result is None:
            return None
        p05 = getattr(pred_result, "monte_carlo_nps_p05", None)
        p50 = getattr(pred_result, "monte_carlo_nps_p50", None)
        p95 = getattr(pred_result, "monte_carlo_nps_p95", None)
        if p05 is None or p50 is None or p95 is None:
            return None
        return {
            "p05": float(p05),
            "p50": float(p50),
            "p95": float(p95),
            "basis": "monte_carlo_survey_score_distribution",
        }

optimize = ReverseOptimizer.optimize
