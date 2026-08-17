"""
Statistics module – pure Python deterministic statistics.
No external dependencies.
"""
import math
from typing import List, Optional

class Statistics:
    @staticmethod
    def mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def median(values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 1:
            return sorted_vals[n // 2]
        else:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    @staticmethod
    def variance(values: List[float], sample: bool = True) -> float:
        if len(values) < 2:
            return 0.0
        mean_val = Statistics.mean(values)
        sq_diffs = [(x - mean_val) ** 2 for x in values]
        if sample:
            return sum(sq_diffs) / (len(values) - 1)
        else:
            return sum(sq_diffs) / len(values)

    @staticmethod
    def std_dev(values: List[float], sample: bool = True) -> float:
        return math.sqrt(Statistics.variance(values, sample))

    @staticmethod
    def slope(values: List[float]) -> float:
        """
        Compute the slope of the linear regression of values over equal time steps.
        Returns normalized slope (slope / mean value).
        """
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = Statistics.mean(values)
        if y_mean == 0:
            return 0.0
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        raw_slope = numerator / denominator
        # Normalize by the mean to get relative change
        return raw_slope / y_mean

    @staticmethod
    def moving_average(values: List[float], window: int = 3) -> List[float]:
        """
        Compute moving average with expanding window for early points.
        For points before the window is full, uses an expanding window.
        After window is full, uses a sliding window.
        """
        if len(values) < window:
            return values[:]
        result = []
        for i in range(len(values)):
            if i < window - 1:
                # Expanding window for early points
                window_vals = values[:i+1]
                result.append(Statistics.mean(window_vals))
            else:
                result.append(Statistics.mean(values[i-window+1:i+1]))
        return result

    @staticmethod
    def absolute_change(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        return values[-1] - values[0]

    @staticmethod
    def percent_change(values: List[float]) -> float:
        if len(values) < 2 or values[0] == 0:
            return 0.0
        return ((values[-1] - values[0]) / values[0]) * 100.0
