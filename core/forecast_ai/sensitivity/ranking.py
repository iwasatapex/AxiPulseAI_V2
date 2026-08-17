"""
SensitivityRanker – sorts analyses by absolute sensitivity.
"""
from typing import List
from .models import SensitivityAnalysis

class SensitivityRanker:
    @staticmethod
    def rank(analyses: List[SensitivityAnalysis]) -> List[SensitivityAnalysis]:
        """Rank by operational sensitivity importance."""

        priority = {
            "quality": 5,
            "competency": 4,
            "attendance": 3,
            "release": 2,
            "transfer": 1,
        }

        sorted_analyses = sorted(
            analyses,
            key=lambda a: priority.get(a.metric, 0),
            reverse=True
        )

        for idx, analysis in enumerate(sorted_analyses, 1):
            analysis.rank = idx

        return sorted_analyses
