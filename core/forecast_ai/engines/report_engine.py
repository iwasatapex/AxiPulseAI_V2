"""
ReportEngine – ForecastAI engine for report generation.
"""
import datetime
import logging
from typing import Optional
from dataclasses import asdict
from ..base_engine import ForecastAIEngine
from ..models import ForecastRequest, ForecastResponse
from ..reporting import ReportEngine as RepCore, ReportType

logger = logging.getLogger(__name__)

class ReportEngine(ForecastAIEngine):
    def __init__(self, core: Optional[RepCore] = None):
        self.core = core or RepCore()

    def execute(self, request: ForecastRequest) -> ForecastResponse:
        if request.parameters is None:
            return self._error_response("Missing parameters")
        p = request.parameters
        report_type = p.get('report_type', 'executive')
        try:
            result = self.core.generate(
                forecast_result=p.get('forecast_result'),
                trend_result=p.get('trend_result'),
                sensitivity_result=p.get('sensitivity_result'),
                recommendation_result=p.get('recommendation_result'),
                strategy_result=p.get('strategy_result'),
                confidence_result=p.get('confidence_result'),
                risk_result=p.get('risk_result'),
                explainability_result=p.get('explainability_result'),
                report_type=report_type
            )
        except Exception as e:
            logger.exception("Report generation failed")
            return self._error_response(f"Report error: {str(e)}")

        payload = {
            "success": result.success,
            "title": result.title,
            "executive_summary": asdict(result.executive_summary) if result.executive_summary else None,
            "sections": [asdict(s) for s in result.sections],
            "metadata": asdict(result.metadata) if result.metadata else None,
            "warnings": result.warnings,
            "errors": result.errors
        }

        return ForecastResponse(
            success=result.success,
            operation="report",
            engine="ReportEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=result.warnings,
            errors=result.errors,
            metadata={"phase": "15", "report_type": report_type},
            payload=payload
        )

    def _error_response(self, message: str) -> ForecastResponse:
        return ForecastResponse(
            success=False,
            operation="report",
            engine="ReportEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[],
            errors=[message],
            metadata={},
            payload=None
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
execute = ReportEngine.execute
