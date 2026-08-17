"""
ConfidenceEngine – ForecastAI engine for confidence evaluation.
Consumes results from other components and produces confidence metrics.
"""
import datetime
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from ..base_engine import ForecastAIEngine
from ..models import ForecastRequest, ForecastResponse
from ..confidence import ConfidenceEngine as ConfCore, ConfidenceResult

logger = logging.getLogger(__name__)

class ConfidenceEngine(ForecastAIEngine):
    def __init__(self, core: Optional[ConfCore] = None):
        self.core = core or ConfCore()

    def execute(self, request: ForecastRequest) -> ForecastResponse:
        if request.parameters is None:
            return self._error_response("Missing parameters")

        # Extract result objects from parameters
        forecast_result = request.parameters.get('forecast_result')
        trend_result = request.parameters.get('trend_result')
        sensitivity_result = request.parameters.get('sensitivity_result')
        recommendation_result = request.parameters.get('recommendation_result')
        strategy_result = request.parameters.get('strategy_result')

        try:
            result = self.core.evaluate(
                forecast_result=forecast_result,
                trend_result=trend_result,
                sensitivity_result=sensitivity_result,
                recommendation_result=recommendation_result,
                strategy_result=strategy_result
            )
        except Exception as e:
            logger.exception("Confidence evaluation failed")
            return self._error_response(f"Confidence error: {str(e)}")

        payload = asdict(result)
        return ForecastResponse(
            success=result.success,
            operation="confidence",
            engine="ConfidenceEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=result.warnings,
            errors=result.errors,
            metadata={"phase": "12", "mode": "analytics"},
            payload=payload
        )

    def _error_response(self, message: str) -> ForecastResponse:
        return ForecastResponse(
            success=False,
            operation="confidence",
            engine="ConfidenceEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[],
            errors=[message],
            metadata={},
            payload=None
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
execute = ConfidenceEngine.execute
