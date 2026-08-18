"""
RecommendationEngine – ForecastAI engine for generating operational advice.
"""
import datetime
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from ..base_engine import ForecastAIEngine
from ..models import ForecastRequest, ForecastResponse
from ..optimization import (
    OptimizationRequest,
    TargetGoal,
    Constraint,
    ConstraintType,
)
from ..optimization import ReverseOptimizer as OptimizerCore
from ..recommendations import RecommendationEngine as RecEngine, RecommendationResult

logger = logging.getLogger(__name__)

class RecommendationEngine(ForecastAIEngine):
    def __init__(self, optimizer_core: Optional[OptimizerCore] = None,
                 rec_engine: Optional[RecEngine] = None):
        self.optimizer = optimizer_core or OptimizerCore()
        self.rec_engine = rec_engine or RecEngine()

    def execute(self, request: ForecastRequest) -> ForecastResponse:
        # Extract parameters
        if request.parameters is None:
            return self._error_response("Missing parameters")

        target_oh = request.parameters.get('target_oh')
        target_nps = request.parameters.get('target_nps')
        if target_oh is None and target_nps is None:
            return self._error_response("Missing 'target_oh' or 'target_nps'")

        state = request.parameters.get('state')
        if state is None:
            return self._error_response("Missing 'state'")

        constraints = []
        for c in request.parameters.get('constraints', []):
            if isinstance(c, dict):
                constraints.append(Constraint(
                    field=c.get('field'),
                    type=ConstraintType(c.get('type')),
                    value=c.get('value'),
                    metadata=c.get('metadata', {})
                ))

        target_goal = TargetGoal(
            target_operations_health=target_oh,
            target_nps=target_nps,
            deadline_days=request.parameters.get('deadline_days'),
            priority=request.parameters.get('priority', 'balanced'),
            tolerance=request.parameters.get('tolerance', 0.5),
            constraints=constraints
        )

        opt_request = OptimizationRequest(
            initial_state=state,
            target_goal=target_goal,
            max_iterations=request.parameters.get('max_iterations', 100),
            timeout_seconds=request.parameters.get('timeout_seconds', 30),
            seed=request.parameters.get('seed')
        )

        # Run optimizer
        try:
            opt_result = self.optimizer.optimize(opt_request)
        except Exception as e:
            logger.exception("Optimization failed")
            return self._error_response(f"Optimization error: {str(e)}")

        # Generate recommendations
        rec_result = self.rec_engine.generate(opt_result)

        best_effort = bool((opt_result.metadata or {}).get("best_effort"))

        # Structured failure reason so callers can distinguish outcomes. A
        # best-effort advisory result is NOT a failure: recommendations are
        # delivered but explicitly marked as target not achieved.
        failure_reason = None
        if not opt_result.success and not best_effort:
            opt_metadata = getattr(opt_result, "metadata", {}) or {}
            if opt_metadata.get("timed_out"):
                failure_reason = "optimization_timeout"
            elif getattr(opt_result, "errors", None):
                failure_reason = "optimization_failed"
            else:
                failure_reason = "insufficient_feasible_actions"

        # Build response
        payload = {
            "optimization": asdict(opt_result),
            "recommendations": {
                "success": rec_result.success,
                "recommendations": [asdict(r) for r in rec_result.recommendations],
                "warnings": rec_result.warnings,
                "errors": rec_result.errors,
                "metadata": rec_result.metadata
            }
        }

        metadata = {
            "phase": "8",
            "goal_achieved": bool(opt_result.success),
            "best_effort": best_effort,
        }
        if best_effort:
            metadata["reason"] = "best_effort"
        if failure_reason:
            metadata["reason"] = failure_reason

        return ForecastResponse(
            success=rec_result.success,
            operation="recommend",
            engine="RecommendationEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=rec_result.warnings,
            errors=rec_result.errors,
            metadata=metadata,
            payload=payload
        )

    def _error_response(self, message: str) -> ForecastResponse:
        return ForecastResponse(
            success=False,
            operation="recommend",
            engine="RecommendationEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[],
            errors=[message],
            metadata={},
            payload=None
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
execute = RecommendationEngine.execute
