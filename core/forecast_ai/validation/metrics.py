
from typing import List


class ForecastMetrics:

    @staticmethod
    def mae(errors: List[float]) -> float:
        if not errors:
            return 0.0

        return sum(
            abs(x) for x in errors
        ) / len(errors)


    @staticmethod
    def bias(errors: List[float]) -> float:
        if not errors:
            return 0.0

        return sum(errors) / len(errors)


    @staticmethod
    def rmse(errors: List[float]) -> float:
        if not errors:
            return 0.0

        return (
            sum(
                x*x for x in errors
            ) / len(errors)
        ) ** 0.5
