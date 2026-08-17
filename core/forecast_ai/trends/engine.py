"""
TrendEngine – orchestrates analysis of multiple KPI timelines.
"""
from typing import List, Dict, Any
from .models import TrendSeries, TrendResult
from .analyzer import TrendAnalyzer

class TrendEngine:
    def __init__(self, window: int = 3):
        self.window = window
        self.analyzer = TrendAnalyzer()

    def analyze(self, series_list: List[TrendSeries]) -> TrendResult:
        if not series_list:
            return TrendResult(
                success=False,
                analyses=[],
                warnings=[],
                errors=["No series provided for analysis."]
            )

        analyses = []
        warnings = []
        errors = []

        for series in series_list:
            if not series.values:
                warnings.append(f"Series '{series.metric}' has no data, skipping.")
                continue
            try:
                analysis = self.analyzer.analyze(series, self.window)
                analyses.append(analysis)
            except Exception as e:
                errors.append(f"Error analyzing '{series.metric}': {str(e)}")

        success = len(analyses) > 0
        return TrendResult(
            success=success,
            analyses=analyses,
            warnings=warnings,
            errors=errors,
            metadata={"total_series": len(series_list), "analyzed": len(analyses)}
        )
