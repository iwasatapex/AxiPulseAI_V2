"""
Pattern detection – deterministic rule-based pattern recognition.
Returns only pattern types: Recovery, Spike, Oscillation, Plateau, Stable.
"""
from typing import List, Tuple
from .statistics import Statistics
from ..config import TREND_THRESHOLDS

class PatternDetector:
    @staticmethod
    def detect(values: List[float]) -> Tuple[str, float]:
        """
        Detect pattern and return (pattern_name, confidence).
        Patterns: Recovery, Spike, Oscillation, Plateau, Stable.
        """
        if len(values) < 3:
            return "Stable", 1.0

        # Compute moving average to smooth noise
        ma = Statistics.moving_average(values, window=3)
        slope = Statistics.slope(ma)  # use smoothed values for slope

        # Check for spike first: isolated extreme values override oscillation.
        mean_val = Statistics.mean(values)
        std_val = Statistics.std_dev(values, sample=False)
        if std_val > 0 and mean_val > 0:
            max_val = max(values)
            if max_val > mean_val + TREND_THRESHOLDS['spike_std_multiplier'] * std_val:
                return "Spike", 0.8

        # Check for oscillation: alternating high/low
        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
        sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i-1] < 0)
        if len(diffs) > 0:
            osc_ratio = sign_changes / len(diffs)
            if osc_ratio > TREND_THRESHOLDS['oscillation_sign_changes']:
                return "Oscillation", min(1.0, osc_ratio + 0.2)

        # Check for recovery: dip then rise
        if len(values) >= 4:
            first_third = Statistics.mean(values[:len(values)//3])
            last_third = Statistics.mean(values[-len(values)//3:])
            if last_third > first_third * TREND_THRESHOLDS['recovery_ratio'] and min(values) < first_third * 0.95:
                return "Recovery", 0.75

        # Check for plateau: flat center movement with no meaningful drift.
        cv = (Statistics.std_dev(values, sample=False) / (Statistics.mean(values) + 1e-6))
        value_range = max(values) - min(values)
        net_change = values[-1] - values[0]

        if cv < TREND_THRESHOLDS['volatility_low'] and value_range < 0.35:
            # Plateau has minimal net movement.
            if abs(net_change) <= 0.25:
                return "Plateau", 0.85

        return "Stable", 0.9
