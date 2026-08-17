"""
TrendEngine – ForecastAI engine for KPI trend analysis.
Consumes timeline data, returns trend analyses.
"""
import datetime
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from ..base_engine import ForecastAIEngine
from ..models import ForecastRequest, ForecastResponse
from ..trends import TrendEngine as TrendCore, TrendSeries, TrendResult

logger = logging.getLogger(__name__)

class TrendEngine(ForecastAIEngine):
    def __init__(self, trend_core: Optional[TrendCore] = None):
        self.trend_core = trend_core or TrendCore()

    def execute(self, request: ForecastRequest) -> ForecastResponse:
        if request.parameters is None:
            return self._error_response("Missing parameters")

        # Expect a list of series in parameters
        series_data = request.parameters.get('series')
        if series_data is None:
            return self._error_response("Missing 'series' in parameters")

        # Build TrendSeries objects
        try:
            series_list = []
            for item in series_data:
                # item should be a dict with metric, values, timestamps
                if not isinstance(item, dict):
                    return self._error_response("Each series item must be a dict")
                metric = item.get('metric')
                values = item.get('values', [])
                timestamps = item.get('timestamps', [])
                if not metric:
                    return self._error_response("Missing 'metric' in series item")
                if not values or not timestamps:
                    continue  # skip empty series
                series_list.append(TrendSeries(
                    metric=metric,
                    values=values,
                    timestamps=timestamps,
                    metadata=item.get('metadata', {})
                ))
        except Exception as e:
            return self._error_response(f"Error parsing series: {str(e)}")

        try:
            result = self.trend_core.analyze(series_list)
        except Exception as e:
            logger.exception("Trend analysis failed")
            return self._error_response(f"Trend analysis error: {str(e)}")

        # Build response
        payload = {
            "analyses": [asdict(a) for a in result.analyses],
            "warnings": result.warnings,
            "errors": result.errors,
            "metadata": result.metadata
        }

        return ForecastResponse(
            success=result.success,
            operation="trend",
            engine="TrendEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=result.warnings,
            errors=result.errors,
            metadata={"phase": "10", "mode": "analytics"},
            payload=payload
        )

    def _error_response(self, message: str) -> ForecastResponse:
        return ForecastResponse(
            success=False,
            operation="trend",
            engine="TrendEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[],
            errors=[message],
            metadata={},
            payload=None
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
execute = TrendEngine.execute
