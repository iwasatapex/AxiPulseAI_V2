"""
ReverseOptimizer – ForecastAI engine for reverse optimization.
"""
import datetime
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from ..base_engine import ForecastAIEngine
from ..models import ForecastRequest, ForecastResponse
from ..optimization import (
    ReverseOptimizer as OptimizerCore,
    TargetGoal,
    Constraint,
    ConstraintType,
    OptimizationRequest
)

logger = logging.getLogger(__name__)

class ReverseOptimizer(ForecastAIEngine):
    def __init__(self, optimizer_core: Optional[OptimizerCore] = None):
        self.core = optimizer_core or OptimizerCore()

    def execute(self, request: ForecastRequest) -> ForecastResponse:
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
            seed=request.parameters.get('seed')  # not used in deterministic search
        )

        try:
            result = self.core.optimize(opt_request)
        except Exception as e:
            logger.exception("Optimization failed")
            return self._error_response(f"Optimization error: {str(e)}")

        payload = asdict(result)
        return ForecastResponse(
            success=result.success,
            operation="reverse_optimize",
            engine="ReverseOptimizer",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=result.warnings,
            errors=result.errors,
            metadata={"phase": "7"},
            payload=payload
        )

    def _error_response(self, message: str) -> ForecastResponse:
        return ForecastResponse(
            success=False,
            operation="reverse_optimize",
            engine="ReverseOptimizer",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[],
            errors=[message],
            metadata={},
            payload=None
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
execute = ReverseOptimizer.execute
