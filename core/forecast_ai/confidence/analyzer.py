"""
ConfidenceAnalyzer – evaluates confidence for each ForecastAI component.
"""
from typing import List, Dict, Any, Optional
from .models import ConfidenceMetric, ConfidenceAnalysis
from .metrics import ConfidenceMetrics
from .scoring import ConfidenceScorer
from ..config import CONFIDENCE_WEIGHTS
from ..models import ForecastResult
from ..trends import TrendAnalysis, TrendResult
from ..sensitivity import SensitivityAnalysis, SensitivityResult
from ..recommendations import RecommendationResult
from ..strategy import StrategyResult

class ConfidenceAnalyzer:
    @staticmethod
    def analyze_forecast(forecast_result: Dict) -> Optional[ConfidenceAnalysis]:
        if not forecast_result:
            return None
        timeline = forecast_result.get('timeline', [])
        if not timeline:
            return None

        metrics = []
        # Prediction stability
        stab_score = ConfidenceMetrics.prediction_stability(timeline)
        metrics.append(ConfidenceMetric(
            name="Prediction Stability",
            score=stab_score,
            weight=CONFIDENCE_WEIGHTS['prediction_stability'],
            reason=f"Stability score {stab_score:.2f} based on forecast volatility."
        ))
        # Combine
        score = ConfidenceScorer.compute_confidence(metrics)
        classification = ConfidenceScorer.classify(score)
        # Detailed reasoning with metric breakdown
        breakdown = ", ".join([f"{m.name}: {m.score:.2f}" for m in metrics])
        reasoning = f"Forecast confidence is {classification.lower()} ({score:.2f}). Metrics: {breakdown}."
        warnings = []
        if stab_score < 0.5:
            warnings.append("Prediction stability is low; forecast may be volatile.")
        return ConfidenceAnalysis(
            component="forecast",
            confidence_score=score,
            classification=classification,
            metrics=metrics,
            reasoning=reasoning,
            warnings=warnings
        )

    @staticmethod
    def analyze_trend(trend_result: TrendResult) -> Optional[ConfidenceAnalysis]:
        if not trend_result or not trend_result.analyses:
            return None
        metrics = []
        # Trend consistency
        cons_score = ConfidenceMetrics.trend_consistency(trend_result.analyses)
        metrics.append(ConfidenceMetric(
            name="Trend Consistency",
            score=cons_score,
            weight=CONFIDENCE_WEIGHTS['trend_consistency'],
            reason=f"Consistency score {cons_score:.2f} based on alignment of KPI directions."
        ))
        score = ConfidenceScorer.compute_confidence(metrics)
        classification = ConfidenceScorer.classify(score)
        breakdown = ", ".join([f"{m.name}: {m.score:.2f}" for m in metrics])
        reasoning = f"Trend confidence is {classification.lower()} ({score:.2f}). Metrics: {breakdown}."
        warnings = []
        if cons_score < 0.5:
            warnings.append("Trend directions are inconsistent across KPIs.")
        return ConfidenceAnalysis(
            component="trend",
            confidence_score=score,
            classification=classification,
            metrics=metrics,
            reasoning=reasoning,
            warnings=warnings
        )

    @staticmethod
    def analyze_sensitivity(sensitivity_result: SensitivityResult) -> Optional[ConfidenceAnalysis]:
        if not sensitivity_result or not sensitivity_result.analyses:
            return None
        metrics = []
        # Sensitivity consistency
        cons_score = ConfidenceMetrics.sensitivity_consistency(sensitivity_result.analyses)
        metrics.append(ConfidenceMetric(
            name="Sensitivity Consistency",
            score=cons_score,
            weight=CONFIDENCE_WEIGHTS['sensitivity_consistency'],
            reason=f"Consistency score {cons_score:.2f} based on differentiation of KPI influence."
        ))
        score = ConfidenceScorer.compute_confidence(metrics)
        classification = ConfidenceScorer.classify(score)
        breakdown = ", ".join([f"{m.name}: {m.score:.2f}" for m in metrics])
        reasoning = f"Sensitivity confidence is {classification.lower()} ({score:.2f}). Metrics: {breakdown}."
        warnings = []
        if cons_score < 0.4:
            warnings.append("Sensitivity signals are weak or inconsistent; KPI influence may be uncertain.")
        return ConfidenceAnalysis(
            component="sensitivity",
            confidence_score=score,
            classification=classification,
            metrics=metrics,
            reasoning=reasoning,
            warnings=warnings
        )

    @staticmethod
    def analyze_recommendations(rec_result: RecommendationResult) -> Optional[ConfidenceAnalysis]:
        if not rec_result or not rec_result.recommendations:
            return None
        metrics = []
        # Recommendation agreement
        agree_score = ConfidenceMetrics.recommendation_agreement(rec_result.recommendations)
        metrics.append(ConfidenceMetric(
            name="Recommendation Agreement",
            score=agree_score,
            weight=CONFIDENCE_WEIGHTS['recommendation_agreement'],
            reason=f"Agreement score {agree_score:.2f} based on category distribution."
        ))
        score = ConfidenceScorer.compute_confidence(metrics)
        classification = ConfidenceScorer.classify(score)
        breakdown = ", ".join([f"{m.name}: {m.score:.2f}" for m in metrics])
        reasoning = f"Recommendation confidence is {classification.lower()} ({score:.2f}). Metrics: {breakdown}."
        warnings = []
        if agree_score < 0.5:
            warnings.append("Recommendations are spread across multiple categories; focus may be unclear.")
        return ConfidenceAnalysis(
            component="recommendation",
            confidence_score=score,
            classification=classification,
            metrics=metrics,
            reasoning=reasoning,
            warnings=warnings
        )

    @staticmethod
    def analyze_strategy(strategy_result: StrategyResult) -> Optional[ConfidenceAnalysis]:
        if not strategy_result or not strategy_result.strategies:
            return None
        metrics = []
        # Strategy completeness
        comp_score = ConfidenceMetrics.strategy_completeness(strategy_result.strategies)
        metrics.append(ConfidenceMetric(
            name="Strategy Completeness",
            score=comp_score,
            weight=CONFIDENCE_WEIGHTS['strategy_completeness'],
            reason=f"Completeness score {comp_score:.2f} based on milestones, risks, and recommendations."
        ))
        score = ConfidenceScorer.compute_confidence(metrics)
        classification = ConfidenceScorer.classify(score)
        breakdown = ", ".join([f"{m.name}: {m.score:.2f}" for m in metrics])
        reasoning = f"Strategy confidence is {classification.lower()} ({score:.2f}). Metrics: {breakdown}."
        warnings = []
        if comp_score < 0.5:
            warnings.append("Strategies lack completeness; milestones or risks may be missing.")
        return ConfidenceAnalysis(
            component="strategy",
            confidence_score=score,
            classification=classification,
            metrics=metrics,
            reasoning=reasoning,
            warnings=warnings
        )

# Module-level compatibility surface.
#
# These aliases intentionally delegate to the existing
# ConfidenceAnalyzer static methods. No confidence logic,
# scoring, thresholds, or result construction is changed.

analyze_forecast = ConfidenceAnalyzer.analyze_forecast
analyze_trend = ConfidenceAnalyzer.analyze_trend
analyze_sensitivity = ConfidenceAnalyzer.analyze_sensitivity
analyze_recommendations = ConfidenceAnalyzer.analyze_recommendations
analyze_strategy = ConfidenceAnalyzer.analyze_strategy
