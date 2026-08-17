"""
StrategyPlanner – groups recommendations into strategy categories.
"""
from typing import List, Dict
from .models import StrategyCategory
from ..recommendations import Recommendation

class StrategyPlanner:
    @staticmethod
    def group_recommendations(recommendations: List[Recommendation]) -> Dict[StrategyCategory, List[str]]:
        """
        Group recommendation IDs by strategy category based on recommendation category.
        """
        grouped = {cat: [] for cat in StrategyCategory}
        for rec in recommendations:
            # Map recommendation category to strategy category
            category_mapping = {
                'quality': StrategyCategory.QUALITY,
                'competency': StrategyCategory.TRAINING,
                'training': StrategyCategory.TRAINING,
                'attendance': StrategyCategory.STAFFING,
                'staffing': StrategyCategory.STAFFING,
                'transfer': StrategyCategory.CUSTOMER_EXPERIENCE,
                'release': StrategyCategory.CUSTOMER_EXPERIENCE,
                'customer_experience': StrategyCategory.CUSTOMER_EXPERIENCE,
                'technology': StrategyCategory.TECHNOLOGY,
                'operations': StrategyCategory.OPERATIONAL_EXCELLENCE,
                'general': StrategyCategory.GENERAL,
                'balanced': StrategyCategory.BALANCED,
                'recovery': StrategyCategory.RECOVERY,
                'preventive': StrategyCategory.PREVENTIVE,
            }
            # Use rec.category.value which is a string
            rec_cat = rec.category.value.lower()
            strategy_cat = category_mapping.get(rec_cat, StrategyCategory.GENERAL)
            grouped[strategy_cat].append(rec.id)
        return grouped
