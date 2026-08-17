"""
Ranking engine for recommendations.
"""
from typing import List
from .models import Recommendation

priority_order = {
    'critical': 0,
    'high': 1,
    'medium': 2,
    'low': 3,
    'informational': 4
}

difficulty_order = {
    'very_easy': 0,
    'easy': 1,
    'medium': 2,
    'hard': 3,
    'very_hard': 4
}


def key_func(r: Recommendation):
    # Tuple: primary key, secondary keys
    return (
        priority_order.get(r.priority.value, 5),
        -(r.estimated_operations_health_gain or 0.0),
        -(r.estimated_nps_gain or 0.0),
        r.optimization_score,
        difficulty_order.get(r.difficulty.value, 5)
    )


class RecommendationRanker:
    @staticmethod
    def rank(recommendations: List[Recommendation]) -> List[Recommendation]:
        """
        Rank recommendations deterministically by:
        1. Priority (critical > high > medium > low > informational)
        2. Estimated OH gain (higher is better)
        3. NPS gain (higher is better)
        4. Optimization score (lower is better)
        5. Difficulty (easier is better)
        """
        return sorted(recommendations, key=key_func)


rank = RecommendationRanker.rank
