import datetime
import logging
from typing import Optional
from dataclasses import asdict
from ..base_engine import ForecastAIEngine
from ..models import ForecastRequest, ForecastResponse
from ..risk import RiskEngine as RiskCore

logger = logging.getLogger(__name__)

class RiskEngine(ForecastAIEngine):
    def __init__(self, core: Optional[RiskCore] = None):
        self.core = core or RiskCore()

    def execute(self, request: ForecastRequest) -> ForecastResponse:
        if request.parameters is None:
            return self._error_response("Missing parameters")
        p = request.parameters
        try:
            result = self.core.evaluate(
                forecast_result=p.get('forecast_result'),
                trend_result=p.get('trend_result'),
                sensitivity_result=p.get('sensitivity_result'),
                recommendation_result=p.get('recommendation_result'),
                strategy_result=p.get('strategy_result'),
                confidence_result=p.get('confidence_result')
            )
        except Exception as e:
            logger.exception("Risk evaluation failed")
            return self._error_response(f"Risk error: {str(e)}")
        return ForecastResponse(
            success=result.success,
            operation="risk",
            engine="RiskEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=result.warnings,
            errors=result.errors,
            metadata={"phase": "13"},
            payload=asdict(result)
        )

    def _error_response(self, message: str) -> ForecastResponse:
        return ForecastResponse(
            success=False,
            operation="risk",
            engine="RiskEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[],
            errors=[message],
            metadata={},
            payload=None
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
execute = RiskEngine.execute
