"""
ConfidenceMetrics – deterministic metric calculators.
Each returns a score between 0.0 and 1.0, and a reason.
"""
import math
from typing import List, Dict, Any, Optional
from ..trends import TrendAnalysis
from ..sensitivity import SensitivityAnalysis
from ..recommendations import Recommendation
from ..strategy import StrategyPlan

def cv(vals):
    if not vals:
        return 1.0
    mean = sum(vals) / len(vals)
    if mean == 0:
        return 1.0
    std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
    return std / mean


class ConfidenceMetrics:
    @staticmethod
    def prediction_stability(forecast_timeline: List[Dict]) -> float:
        """
        Measure stability of predictions across the forecast horizon.
        Low volatility → high confidence.
        """
        if not forecast_timeline:
            return 0.0
        # Extract OH and NPS values
        oh_vals = [day.get('operations_health', 0) for day in forecast_timeline if day.get('operations_health') is not None]
        nps_vals = [day.get('nps', 0) for day in forecast_timeline if day.get('nps') is not None]
        if not oh_vals and not nps_vals:
            return 0.0
        # Compute coefficient of variation for each
        def cv(vals):
            if not vals:
                return 1.0
            mean = sum(vals)/len(vals)
            if mean == 0:
                return 1.0
            std = (sum((x-mean)**2 for x in vals)/len(vals))**0.5
            return std / mean
        oh_cv = cv(oh_vals) if oh_vals else 1.0
        nps_cv = cv(nps_vals) if nps_vals else 1.0
        avg_cv = (oh_cv + nps_cv) / 2.0
        # Convert CV to confidence: lower CV -> higher confidence
        # CV of 0 -> confidence 1.0, CV of 0.5 -> 0.5, cap at 1.0
        conf = max(0.0, min(1.0, 1.0 - avg_cv * 2.0))
        return conf

    @staticmethod
    def trend_consistency(trend_analyses: List[TrendAnalysis]) -> float:
        """
        Consistency of trend directions across KPIs.
        """
        if not trend_analyses:
            return 0.0
        # Count how many KPIs show improvement vs decline
        improving = sum(1 for a in trend_analyses if a.trend_direction in ['Increase', 'Strong Increase'])
        declining = sum(1 for a in trend_analyses if a.trend_direction in ['Decrease', 'Strong Decrease'])
        total = len(trend_analyses)
        if total == 0:
            return 0.0
        # High consistency if most KPIs are either improving or declining together
        max_dir = max(improving, declining)
        consistency = max_dir / total
        # Also factor in strength (if strong trends are more consistent)
        strength_factor = sum(1 for a in trend_analyses if a.trend_strength in ['Strong', 'Moderate']) / total
        return (consistency * 0.7 + strength_factor * 0.3)

    @staticmethod
    def sensitivity_consistency(sensitivity_analyses: List[SensitivityAnalysis]) -> float:
        """
        Consistency of sensitivity rankings.
        """
        if not sensitivity_analyses:
            return 0.0
        # Check if sensitivity scores are reasonably different (not all equal)
        scores = [abs(a.sensitivity_score_oh) for a in sensitivity_analyses]
        if not scores:
            return 0.0
        # If all scores are zero, confidence is low
        if all(s == 0 for s in scores):
            return 0.0
        # Consistency: standard deviation relative to mean
        mean = sum(scores)/len(scores)
        if mean == 0:
            return 0.0
        std = (sum((s-mean)**2 for s in scores)/len(scores))**0.5
        # If std/mean is high, it means clear differentiation -> higher confidence
        ratio = std / mean
        # Cap at 1.0
        return min(1.0, ratio * 0.5 + 0.5)

    @staticmethod
    def recommendation_agreement(recommendations: List[Recommendation]) -> float:
        """
        Agreement among recommendations.

        Uses the existing conflict detector and a category-consistency metric.
        Conflicts (e.g. two recommendations driving the same KPI in opposite
        directions) lower the agreement score; recommendations concentrated in
        a single category raise it.
        """
        if not recommendations:
            return 0.0
        # Category distribution (consistency component).
        categories = {}
        for rec in recommendations:
            cat = rec.category.value
            categories[cat] = categories.get(cat, 0) + 1
        if len(categories) == 1:
            category_score = 1.0
        else:
            top = max(categories.values())
            total = len(recommendations)
            category_penalty = 1.0 - (len(categories) - 1) * 0.1
            category_score = max(0.0, (top / total) * max(0.5, category_penalty))

        # Conflict component: use the existing ConflictDetector.
        from ..recommendations.conflicts import ConflictDetector
        conflicts = ConflictDetector.detect_conflicts(recommendations)
        conflict_penalty = min(1.0, len(conflicts) * 0.5)

        # Agreement = category consistency reduced by conflicts.
        return max(0.0, category_score * (1.0 - conflict_penalty))

    @staticmethod
    def strategy_completeness(strategies: List[StrategyPlan]) -> float:
        """
        Completeness of strategies using gradual scoring.
        - Recommendations: 40%
        - Milestones: 40%
        - Risks: 20%
        """
        if not strategies:
            return 0.0
        scores = []
        for s in strategies:
            rec_score = 1.0 if s.recommendations else 0.0
            # Milestones: if both timeline and milestones exist, full score
            milestone_score = 1.0 if (s.timeline or s.milestones) else 0.0
            risk_score = 1.0 if s.risks else 0.0
            total = rec_score * 0.4 + milestone_score * 0.4 + risk_score * 0.2
            scores.append(total)
        return sum(scores) / len(scores)

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
prediction_stability = ConfidenceMetrics.prediction_stability

# Module-level compatibility surface.
# Delegates to existing implementations; no logic changed.
trend_consistency = ConfidenceMetrics.trend_consistency
sensitivity_consistency = ConfidenceMetrics.sensitivity_consistency
recommendation_agreement = ConfidenceMetrics.recommendation_agreement
strategy_completeness = ConfidenceMetrics.strategy_completeness
