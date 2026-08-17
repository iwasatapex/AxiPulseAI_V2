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

        # Setup evaluator: apply scenarios, then call PredictionService
        def evaluator(state: OperationalState) -> tuple:
            # Apply scenarios (if any are active for day 1; we assume day 1 for simplicity)
            # In future, we could pass a day parameter.
            modified_state = self.scenario_manager.apply_scenarios(state, day=1)
            # Predict
            pred_req = PredictionRequest(state=modified_state.to_dict(), metadata={"optimizer": True})
            result = self.service.predict(pred_req)
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

        acceptable = [s for s in solutions if s.distance_to_target <= target.tolerance]
        original_solution = next(
            (s for s in solutions if (s.metadata or {}).get("is_original")),
            None,
        )
        if acceptable:
            best_solution = acceptable[0]  # already sorted (best score first)
            success = True
            best_effort = False
        else:
            best_solution = solutions[0] if solutions else None
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

        if timed_out:
            errors = [
                "optimization_timeout: search exceeded the configured "
                "timeout budget and was aborted"
            ]
            success = False
            best_effort = False
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
            }
        )

optimize = ReverseOptimizer.optimize
