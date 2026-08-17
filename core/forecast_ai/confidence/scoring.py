"""
ConfidenceScorer – combines metrics using configurable weights.
"""
from typing import List
from .models import ConfidenceMetric
from ..config import CONFIDENCE_WEIGHTS, CONFIDENCE_THRESHOLDS

class ConfidenceScorer:
    @staticmethod
    def compute_confidence(metrics: List[ConfidenceMetric]) -> float:
        """Weighted sum of metric scores."""
        if not metrics:
            return 0.0
        total_weight = sum(m.weight for m in metrics)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(m.score * m.weight for m in metrics)
        return weighted_sum / total_weight

    @staticmethod
    def classify(score: float) -> str:
        """Map score to classification using config thresholds."""
        if score >= CONFIDENCE_THRESHOLDS['very_high']:
            return "Very High"
        elif score >= CONFIDENCE_THRESHOLDS['high']:
            return "High"
        elif score >= CONFIDENCE_THRESHOLDS['medium']:
            return "Medium"
        elif score >= CONFIDENCE_THRESHOLDS['low']:
            return "Low"
        else:
            return "Very Low"

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
compute_confidence = ConfidenceScorer.compute_confidence

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
classify = ConfidenceScorer.classify
