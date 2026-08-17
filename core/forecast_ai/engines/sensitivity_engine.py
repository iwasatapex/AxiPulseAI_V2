"""
SensitivityEngine – ForecastAI engine for sensitivity analysis.
Consumes operational state, returns ranked KPI sensitivities.
"""
import datetime
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from ..base_engine import ForecastAIEngine
from ..models import ForecastRequest, ForecastResponse
from ..sensitivity import SensitivityEngine as SenCore, SensitivityResult

logger = logging.getLogger(__name__)

class SensitivityEngine(ForecastAIEngine):
    def __init__(self, sen_core: Optional[SenCore] = None):
        self.sen_core = sen_core or SenCore()

    def execute(self, request: ForecastRequest) -> ForecastResponse:
        if request.parameters is None:
            return self._error_response("Missing parameters")

        state = request.parameters.get('state')
        if state is None:
            return self._error_response("Missing 'state' in parameters")

        step_size = request.parameters.get('step_size', 1.0)
        metrics = request.parameters.get('metrics')

        try:
            result = self.sen_core.analyze(state, metrics)
        except Exception as e:
            logger.exception("Sensitivity analysis failed")
            return self._error_response(f"Sensitivity error: {str(e)}")

        payload = {
            "analyses": [asdict(a) for a in result.analyses],
            "ranking": [asdict(a) for a in result.ranking],
            "warnings": result.warnings,
            "errors": result.errors,
            "metadata": result.metadata
        }

        return ForecastResponse(
            success=result.success,
            operation="sensitivity",
            engine="SensitivityEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=result.warnings,
            errors=result.errors,
            metadata={"phase": "11", "mode": "analytics"},
            payload=payload
        )

    def _error_response(self, message: str) -> ForecastResponse:
        return ForecastResponse(
            success=False,
            operation="sensitivity",
            engine="SensitivityEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[],
            errors=[message],
            metadata={},
            payload=None
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
execute = SensitivityEngine.execute
