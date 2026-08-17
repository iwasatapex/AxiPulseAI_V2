import hashlib
from typing import List, Dict, Any
from ..trends import TrendResult
from ..sensitivity import SensitivityResult
from ..recommendations import RecommendationResult
from ..strategy import StrategyResult
from ..confidence import ConfidenceResult
from .models import RiskFactor, RiskCategory
from .scoring import RiskScorer
from ..config import RISK_DETECTOR_THRESHOLDS

class ForecastRiskDetector:
    @staticmethod
    def detect(forecast_result: Dict) -> List[RiskFactor]:
        risks = []
        if not forecast_result:
            return risks
        timeline = forecast_result.get('timeline', [])
        if not timeline:
            return risks
        oh_vals = [d.get('operations_health', 0) for d in timeline if d.get('operations_health') is not None]
        if len(oh_vals) > 1:
            mean = sum(oh_vals)/len(oh_vals)
            if mean > 0:
                std = (sum((x-mean)**2 for x in oh_vals)/len(oh_vals))**0.5
                cv = std / mean
                if cv > RISK_DETECTOR_THRESHOLDS['forecast_cv_threshold']:
                    risks.append(RiskFactor(
                        id=hashlib.md5("forecast_volatility".encode()).hexdigest()[:8],
                        name="High Forecast Volatility",
                        category=RiskCategory.FORECAST,
                        severity=0.7, probability=0.6, impact=0.5,
                        risk_score=RiskScorer.compute_risk_score(0.7, 0.6, 0.5),
                        reason=f"CV={cv:.2f}",
                        mitigation="Consider smoothing forecast.",
                        source="ForecastRiskDetector",
                        source_kind="business_rule"
                    ))
        if len(oh_vals) >= 2:
            if oh_vals[-1] < oh_vals[0] * RISK_DETECTOR_THRESHOLDS['forecast_degradation_threshold']:
                risks.append(RiskFactor(
                    id=hashlib.md5("forecast_degradation".encode()).hexdigest()[:8],
                    name="Forecast Degradation",
                    category=RiskCategory.FORECAST,
                    severity=0.6, probability=0.7, impact=0.4,
                    risk_score=RiskScorer.compute_risk_score(0.6, 0.7, 0.4),
                    reason="OH declines over time.",
                    mitigation="Investigate root causes.",
                    source="ForecastRiskDetector",
                    source_kind="business_rule"
                ))
        return risks

class TrendRiskDetector:
    @staticmethod
    def detect(trend_result: TrendResult) -> List[RiskFactor]:
        risks = []
        if not trend_result or not trend_result.analyses:
            return risks
        for a in trend_result.analyses:
            if a.volatility in ["High", "Medium"] and a.trend_direction in ["Decrease", "Strong Decrease"]:
                risks.append(RiskFactor(
                    id=hashlib.md5(f"trend_decline_{a.metric}".encode()).hexdigest()[:8],
                    name=f"Decline in {a.metric}",
                    category=RiskCategory.TREND,
                    severity=0.7, probability=0.6, impact=0.5,
                    risk_score=RiskScorer.compute_risk_score(0.7, 0.6, 0.5),
                    reason=f"{a.metric} declining with {a.volatility} volatility.",
                    mitigation=f"Correct {a.metric}.",
                    source="TrendRiskDetector",
                    source_kind="business_rule"
                ))
            elif a.trend_direction == "Strong Decrease":
                risks.append(RiskFactor(
                    id=hashlib.md5(f"strong_decrease_{a.metric}".encode()).hexdigest()[:8],
                    name=f"Strong Decrease in {a.metric}",
                    category=RiskCategory.TREND,
                    severity=0.8, probability=0.7, impact=0.6,
                    risk_score=RiskScorer.compute_risk_score(0.8, 0.7, 0.6),
                    reason=f"{a.metric} shows strong decrease.",
                    mitigation=f"Prioritize {a.metric} intervention.",
                    source="TrendRiskDetector",
                    source_kind="business_rule"
                ))
            if a.pattern == "Oscillation":
                risks.append(RiskFactor(
                    id=hashlib.md5(f"oscillation_{a.metric}".encode()).hexdigest()[:8],
                    name=f"Oscillation in {a.metric}",
                    category=RiskCategory.TREND,
                    severity=0.5, probability=0.5, impact=0.5,
                    risk_score=RiskScorer.compute_risk_score(0.5, 0.5, 0.5),
                    reason=f"{a.metric} oscillates.",
                    mitigation="Stabilize operations.",
                    source="TrendRiskDetector",
                    source_kind="business_rule"
                ))
        return risks

class SensitivityRiskDetector:
    @staticmethod
    def detect(sensitivity_result: SensitivityResult) -> List[RiskFactor]:
        risks = []
        if not sensitivity_result or not sensitivity_result.analyses:
            return risks
        if sensitivity_result.analyses:
            top = sensitivity_result.analyses[0]
            if top.sensitivity_score_oh > RISK_DETECTOR_THRESHOLDS['sensitivity_high_threshold']:
                risks.append(RiskFactor(
                    id=hashlib.md5(f"dep_{top.metric}".encode()).hexdigest()[:8],
                    name=f"High Dependency on {top.metric.title()}",
                    category=RiskCategory.SENSITIVITY,
                    severity=0.6, probability=0.5, impact=0.7,
                    risk_score=RiskScorer.compute_risk_score(0.6, 0.5, 0.7),
                    reason=f"OH sensitive to {top.metric}.",
                    mitigation=f"Reduce reliance on {top.metric}.",
                    source="SensitivityRiskDetector",
                    source_kind="business_rule"
                ))
            weak = sum(1 for a in sensitivity_result.analyses if abs(a.sensitivity_score_oh) < RISK_DETECTOR_THRESHOLDS['sensitivity_weak_threshold'])
            if weak == len(sensitivity_result.analyses):
                risks.append(RiskFactor(
                    id=hashlib.md5("sensitivity_weak".encode()).hexdigest()[:8],
                    name="Weak Sensitivity Signal",
                    category=RiskCategory.SENSITIVITY,
                    severity=0.3, probability=0.4, impact=0.4,
                    risk_score=RiskScorer.compute_risk_score(0.3, 0.4, 0.4),
                    reason="No KPI shows significant influence.",
                    mitigation="Collect more data.",
                    source="SensitivityRiskDetector",
                    source_kind="business_rule"
                ))
        return risks

class RecommendationRiskDetector:
    @staticmethod
    def detect(rec_result: RecommendationResult) -> List[RiskFactor]:
        risks = []
        if not rec_result or not rec_result.recommendations:
            return risks
        recs = rec_result.recommendations
        if len(recs) > RISK_DETECTOR_THRESHOLDS['recommendation_overload_threshold']:
            risks.append(RiskFactor(
                id=hashlib.md5("overload".encode()).hexdigest()[:8],
                name="Recommendation Overload",
                category=RiskCategory.RECOMMENDATION,
                severity=0.5, probability=0.6, impact=0.5,
                risk_score=RiskScorer.compute_risk_score(0.5, 0.6, 0.5),
                reason=f"{len(recs)} recommendations.",
                mitigation=f"Prioritize top {RISK_DETECTOR_THRESHOLDS['recommendation_overload_threshold']}.",
                source="RecommendationRiskDetector",
                source_kind="business_rule"
            ))
        cats = set(r.category for r in recs)
        if len(cats) > RISK_DETECTOR_THRESHOLDS['recommendation_scatter_threshold']:
            risks.append(RiskFactor(
                id=hashlib.md5("scatter".encode()).hexdigest()[:8],
                name="Scattered Recommendations",
                category=RiskCategory.RECOMMENDATION,
                severity=0.4, probability=0.5, impact=0.5,
                risk_score=RiskScorer.compute_risk_score(0.4, 0.5, 0.5),
                reason=f"{len(cats)} categories.",
                mitigation="Consolidate focus.",
                source="RecommendationRiskDetector",
                source_kind="business_rule"
            ))
        return risks

class StrategyRiskDetector:
    @staticmethod
    def detect(strategy_result: StrategyResult) -> List[RiskFactor]:
        risks = []
        if not strategy_result or not strategy_result.strategies:
            return risks
        for s in strategy_result.strategies:
            if s.estimated_complexity > RISK_DETECTOR_THRESHOLDS['strategy_complexity_threshold']:
                risks.append(RiskFactor(
                    id=hashlib.md5(f"complex_{s.id}".encode()).hexdigest()[:8],
                    name=f"High Complexity: {s.name}",
                    category=RiskCategory.STRATEGY,
                    severity=0.6, probability=0.6, impact=0.6,
                    risk_score=RiskScorer.compute_risk_score(0.6, 0.6, 0.6),
                    reason=f"Complexity {s.estimated_complexity:.2f}.",
                    mitigation="Break into phases.",
                    source="StrategyRiskDetector",
                    source_kind="business_rule"
                ))
            if s.estimated_duration_weeks > RISK_DETECTOR_THRESHOLDS['strategy_duration_threshold']:
                risks.append(RiskFactor(
                    id=hashlib.md5(f"duration_{s.id}".encode()).hexdigest()[:8],
                    name=f"Long Duration: {s.name}",
                    category=RiskCategory.STRATEGY,
                    severity=0.4, probability=0.5, impact=0.5,
                    risk_score=RiskScorer.compute_risk_score(0.4, 0.5, 0.5),
                    reason=f"{s.estimated_duration_weeks} weeks.",
                    mitigation="Set milestones.",
                    source="StrategyRiskDetector",
                    source_kind="business_rule"
                ))
            if not s.risks:
                risks.append(RiskFactor(
                    id=hashlib.md5(f"norisk_{s.id}".encode()).hexdigest()[:8],
                    name=f"Missing Risks: {s.name}",
                    category=RiskCategory.STRATEGY,
                    severity=0.5, probability=0.3, impact=0.7,
                    risk_score=RiskScorer.compute_risk_score(0.5, 0.3, 0.7),
                    reason="No risks listed.",
                    mitigation="Assess strategy risks.",
                    source="StrategyRiskDetector",
                    source_kind="business_rule"
                ))
        return risks

class ConfidenceRiskDetector:
    @staticmethod
    def detect(confidence_result: ConfidenceResult) -> List[RiskFactor]:
        risks = []
        if not confidence_result or not confidence_result.analyses:
            return risks
        low_threshold = RISK_DETECTOR_THRESHOLDS['confidence_low_threshold']
        for a in confidence_result.analyses:
            if a.confidence_score < low_threshold:
                risks.append(RiskFactor(
                    id=hashlib.md5(f"lowconf_{a.component}".encode()).hexdigest()[:8],
                    name=f"Low {a.component.title()} Confidence",
                    category=RiskCategory.CONFIDENCE,
                    severity=0.6, probability=0.7, impact=0.5,
                    risk_score=RiskScorer.compute_risk_score(0.6, 0.7, 0.5),
                    reason=f"Score {a.confidence_score:.2f}.",
                    mitigation="Improve data or run more analyses.",
                    source="ConfidenceRiskDetector",
                    source_kind="business_rule"
                ))
            if a.warnings:
                risks.append(RiskFactor(
                    id=hashlib.md5(f"warn_{a.component}".encode()).hexdigest()[:8],
                    name=f"Warnings in {a.component.title()}",
                    category=RiskCategory.CONFIDENCE,
                    severity=0.4, probability=0.5, impact=0.4,
                    risk_score=RiskScorer.compute_risk_score(0.4, 0.5, 0.4),
                    reason=f"{len(a.warnings)} warnings.",
                    mitigation="Address underlying issues.",
                    source="ConfidenceRiskDetector",
                    source_kind="business_rule"
                ))
        return risks


# ---------------------------------------------------------------------------
# Module-level compatibility surface
#
# The detector classes each expose a static ``detect`` for a specific input
# kind (forecast dict, TrendResult, SensitivityResult, RecommendationResult,
# StrategyResult, ConfidenceResult). The module-level ``detect`` dispatches to
# the matching detector based on the input's type.
# ---------------------------------------------------------------------------
def detect(input_value):
    if isinstance(input_value, dict):
        return ForecastRiskDetector.detect(input_value)
    if isinstance(input_value, TrendResult):
        return TrendRiskDetector.detect(input_value)
    if isinstance(input_value, SensitivityResult):
        return SensitivityRiskDetector.detect(input_value)
    if isinstance(input_value, RecommendationResult):
        return RecommendationRiskDetector.detect(input_value)
    if isinstance(input_value, StrategyResult):
        return StrategyRiskDetector.detect(input_value)
    if isinstance(input_value, ConfidenceResult):
        return ConfidenceRiskDetector.detect(input_value)
    raise TypeError(
        f"Unsupported risk input type: {type(input_value).__name__}"
    )
