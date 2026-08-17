
from typing import List, Dict, Any

from .metrics import ForecastMetrics


class ForecastBacktester:

    def evaluate(
        self,
        forecast_records: List[Dict[str, Any]]
    ):

        results = {}

        for metric in [
            "quality",
            "competency",
            "attendance",
            "release",
            "transfer",
            "operations_health",
            "nps",
        ]:

            errors = []

            for item in forecast_records:
                if (
                    item.get("predicted") is not None
                    and item.get("actual") is not None
                    and item.get("metric") == metric
                ):
                    errors.append(
                        item["actual"]
                        -
                        item["predicted"]
                    )

            results[metric] = {
                "mae":
                    ForecastMetrics.mae(errors),
                "rmse":
                    ForecastMetrics.rmse(errors),
                "bias":
                    ForecastMetrics.bias(errors),
                "samples":
                    len(errors)
            }

        return results
