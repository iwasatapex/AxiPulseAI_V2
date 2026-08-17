"""
Strategy scorer – deterministic scoring of strategies.
"""
from typing import List
from .models import StrategyPlan

class StrategyScorer:
    @staticmethod
    def score(strategy: StrategyPlan) -> float:
        """
        Compute a lower-is-better score:
        - Lower disruption
        - Lower complexity
        - Higher confidence
        - Shorter duration
        - Higher priority (Critical=0, High=1, ...)
        """
        priority_map = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        priority_score = priority_map.get(strategy.priority, 2)
        disruption = strategy.estimated_disruption
        complexity = strategy.estimated_complexity
        confidence = 1.0 - strategy.confidence  # lower confidence = higher score
        duration = strategy.estimated_duration_weeks / 10.0  # normalize

        return (
            priority_score * 0.3 +
            disruption * 0.25 +
            complexity * 0.2 +
            confidence * 0.15 +
            duration * 0.1
        )

    @staticmethod
    def rank(strategies: List[StrategyPlan]) -> List[StrategyPlan]:
        """Rank strategies by score (lower is better)."""
        return sorted(strategies, key=lambda s: StrategyScorer.score(s))
