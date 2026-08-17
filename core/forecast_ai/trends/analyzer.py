"""
TrendAnalyzer – analyzes a single KPI series and produces TrendAnalysis.
"""
from typing import List
from .models import TrendSeries, TrendAnalysis
from .statistics import Statistics
from .patterns import PatternDetector
from ..config import TREND_THRESHOLDS

class TrendAnalyzer:
    @staticmethod
    def analyze(series: TrendSeries, window: int = 3) -> TrendAnalysis:
        values = series.values
        if not values:
            return TrendAnalysis(
                metric=series.metric,
                trend_direction="Stable",
                trend_strength="Weak",
                moving_average=[],
                minimum=0.0,
                maximum=0.0,
                mean=0.0,
                median=0.0,
                variance=0.0,
                standard_deviation=0.0,
                volatility="Low",
                absolute_change=0.0,
                percent_change=0.0,
                pattern="Stable",
                confidence=1.0
            )

        # Compute moving average
        ma = Statistics.moving_average(values, window)

        # Compute slope using the smoothed moving average
        slope = Statistics.slope(ma) * Statistics.mean(ma)

        # Determine trend direction and strength from normalized slope
        abs_slope = abs(slope)
        if abs_slope > TREND_THRESHOLDS['strong_slope']:
            direction = "Strong Increase" if slope > 0 else "Strong Decrease"
            strength = "Strong"
        elif abs_slope > TREND_THRESHOLDS['moderate_slope']:
            direction = "Increase" if slope > 0 else "Decrease"
            strength = "Moderate"
        else:
            direction = "Stable"
            strength = "Weak"

        # Volatility based on coefficient of variation
        mean_val = Statistics.mean(values)
        std = Statistics.std_dev(values, sample=False)
        if mean_val > 0:
            cv = std / mean_val
            if cv < TREND_THRESHOLDS['volatility_low']:
                volatility = "Low"
            elif cv < TREND_THRESHOLDS['volatility_medium']:
                volatility = "Medium"
            else:
                volatility = "High"
        else:
            volatility = "Medium"

        # Pattern detection (pure pattern, no trend)
        pattern, confidence = PatternDetector.detect(values)

        # Oscillating series do not have a directional trend.
        if pattern == "Oscillation":
            direction = "Stable"
            strength = "Weak"

        # Adjust confidence based on data length and consistency
        data_points = len(values)
        length_factor = min(1.0, data_points / 10.0)
        std_factor = 1.0 / (1.0 + std / (mean_val + 1e-6))
        conf = (length_factor * 0.6 + std_factor * 0.4) * confidence
        confidence = min(1.0, conf)

        return TrendAnalysis(
            metric=series.metric,
            trend_direction=direction,
            trend_strength=strength,
            moving_average=ma,
            minimum=min(values) if values else 0.0,
            maximum=max(values) if values else 0.0,
            mean=Statistics.mean(values),
            median=Statistics.median(values),
            variance=Statistics.variance(values, sample=False),
            standard_deviation=Statistics.std_dev(values, sample=False),
            volatility=volatility,
            absolute_change=Statistics.absolute_change(values),
            percent_change=Statistics.percent_change(values),
            pattern=pattern,
            confidence=confidence,
            metadata={"window": window}
        )
