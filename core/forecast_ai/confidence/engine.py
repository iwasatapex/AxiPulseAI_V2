"""
ConfidenceEngine – orchestrates confidence analysis for all ForecastAI components.
"""
import logging
from typing import Dict, Any, Optional
from .models import ConfidenceResult
from .analyzer import ConfidenceAnalyzer
from .scoring import ConfidenceScorer
from ..config import COMPONENT_WEIGHTS

logger = logging.getLogger(__name__)

class ConfidenceEngine:
    def __init__(self):
        self.analyzer = ConfidenceAnalyzer()
        self.scorer = ConfidenceScorer()

    def evaluate(self,
                 forecast_result: Optional[Dict] = None,
                 trend_result: Optional[Any] = None,
                 sensitivity_result: Optional[Any] = None,
                 recommendation_result: Optional[Any] = None,
                 strategy_result: Optional[Any] = None) -> ConfidenceResult:
        """
        Evaluate confidence for each available component.
        All parameters are optional; each is analyzed if provided.
        """
        analyses = []
        warnings = []
        errors = []
        forecast_analysis = None
        trend_analysis = None
        sensitivity_analysis = None
        recommendation_analysis = None
        strategy_analysis = None

        # Map component names to their results and analyzer methods
        components = {
            'forecast': (forecast_result, self.analyzer.analyze_forecast),
            'trend': (trend_result, self.analyzer.analyze_trend),
            'sensitivity': (sensitivity_result, self.analyzer.analyze_sensitivity),
            'recommendation': (recommendation_result, self.analyzer.analyze_recommendations),
            'strategy': (strategy_result, self.analyzer.analyze_strategy),
        }

        component_analyses = {}

        for name, (result, analyzer_func) in components.items():
            if result is not None:
                try:
                    analysis = analyzer_func(result)
                    if analysis:
                        component_analyses[name] = analysis
                        analyses.append(analysis)
                        warnings.extend(analysis.warnings)
                except Exception as e:
                    errors.append(f"{name.title()} confidence error: {str(e)}")

        if not analyses:
            return ConfidenceResult(
                success=False,
                overall_confidence=0.0,
                warnings=warnings,
                errors=errors or ["No analyses could be performed."]
            )

        # Compute weighted overall confidence
        total_weight = 0.0
        weighted_sum = 0.0
        for name, weight in COMPONENT_WEIGHTS.items():
            if name in component_analyses:
                weighted_sum += component_analyses[name].confidence_score * weight
                total_weight += weight

        if total_weight == 0:
            overall = 0.0
        else:
            overall = weighted_sum / total_weight

        # Build result
        result = ConfidenceResult(
            success=True,
            overall_confidence=overall,
            forecast_confidence=component_analyses.get('forecast'),
            trend_confidence=component_analyses.get('trend'),
            sensitivity_confidence=component_analyses.get('sensitivity'),
            recommendation_confidence=component_analyses.get('recommendation'),
            strategy_confidence=component_analyses.get('strategy'),
            analyses=analyses,
            warnings=warnings,
            errors=errors,
            metadata={
                "num_components": len(analyses),
                "component_weights": COMPONENT_WEIGHTS
            }
        )
        return result

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
evaluate = ConfidenceEngine.evaluate
