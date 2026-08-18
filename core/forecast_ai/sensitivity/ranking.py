"""
SensitivityRanker – sorts analyses by absolute sensitivity.
"""
from typing import List
from .models import SensitivityAnalysis


class SensitivityRanker:
    @staticmethod
    def rank(analyses: List[SensitivityAnalysis]) -> List[SensitivityAnalysis]:
        """Rank by actual absolute model sensitivity."""
        sorted_analyses = sorted(
            analyses,
            key=lambda a: abs(float(a.sensitivity_score_oh or 0.0)),
            reverse=True,
        )

        for idx, analysis in enumerate(sorted_analyses, 1):
            analysis.rank = idx

        return sorted_analyses
