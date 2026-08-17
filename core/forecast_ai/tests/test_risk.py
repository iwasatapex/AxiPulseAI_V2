import unittest
from core.forecast_ai.risk import (
    RiskEngine, RiskResult, RiskFactor, RiskAnalysis,
    ForecastRiskDetector, TrendRiskDetector,
    SensitivityRiskDetector, RecommendationRiskDetector,
    StrategyRiskDetector, ConfidenceRiskDetector,
    RiskScorer, RiskFormatter, RiskCategory
)
from core.forecast_ai.trends import TrendAnalysis, TrendResult
from core.forecast_ai.sensitivity import SensitivityAnalysis, SensitivityResult
from core.forecast_ai.recommendations import Recommendation, Category, Priority, Difficulty, RecommendationResult
from core.forecast_ai.strategy import StrategyPlan, StrategyCategory, StrategyResult
from core.forecast_ai.confidence import ConfidenceResult, ConfidenceAnalysis, ConfidenceMetric
from core.forecast_ai.config import COMPONENT_RISK_AGGREGATION

class TestRisk(unittest.TestCase):
    def test_risk_score_weighted(self):
        s = RiskScorer.compute_risk_score(0.7, 0.7, 0.7)
        self.assertAlmostEqual(s, 0.7, places=2)

    def test_classification(self):
        self.assertEqual(RiskScorer.classify(0.8), "Critical")
        self.assertEqual(RiskScorer.classify(0.6), "High")
        self.assertEqual(RiskScorer.classify(0.4), "Medium")
        self.assertEqual(RiskScorer.classify(0.2), "Low")
        self.assertEqual(RiskScorer.classify(0.1), "Very Low")

    def test_forecast_detector(self):
        forecast = {"timeline": [{"operations_health": 100}, {"operations_health": 50}, {"operations_health": 30}]}
        risks = ForecastRiskDetector.detect(forecast)
        self.assertTrue(any("Volatility" in r.name for r in risks))

    def test_trend_detector(self):
        a = TrendAnalysis(metric='OH', trend_direction='Strong Decrease', trend_strength='Strong',
                          moving_average=[], minimum=0, maximum=0, mean=0, median=0,
                          variance=0, standard_deviation=0, volatility='Low',
                          absolute_change=0, percent_change=0, pattern='Decreasing', confidence=0)
        result = TrendResult(success=True, analyses=[a])
        risks = TrendRiskDetector.detect(result)
        self.assertTrue(any("Strong Decrease" in r.name for r in risks))

    def test_sensitivity_detector(self):
        a = SensitivityAnalysis(metric='quality', baseline_output_oh=80, baseline_output_nps=70,
                                modified_output_oh=90, modified_output_nps=75,
                                operations_health_change=10, nps_change=5,
                                sensitivity_score_oh=2.0, sensitivity_score_nps=1.0,
                                elasticity_oh=0.5, elasticity_nps=0.4, rank=0, classification='High')
        result = SensitivityResult(success=True, analyses=[a])
        risks = SensitivityRiskDetector.detect(result)
        self.assertTrue(any("Dependency" in r.name for r in risks))

    def test_recommendation_detector(self):
        recs = [Recommendation(id=str(i), title='A', description='', category=Category.QUALITY,
                               priority=Priority.HIGH, difficulty=Difficulty.EASY) for i in range(6)]
        result = RecommendationResult(success=True, recommendations=recs)
        risks = RecommendationRiskDetector.detect(result)
        self.assertTrue(any("Overload" in r.name for r in risks))

    def test_strategy_detector(self):
        s = StrategyPlan(id='1', name='A', description='', objective='', category=StrategyCategory.GENERAL,
                         priority='High', estimated_duration_weeks=10, estimated_complexity=0.9,
                         estimated_disruption=0.3, confidence=0.7, recommendations=[],
                         timeline=[], milestones=[], risks=[])
        result = StrategyResult(success=True, strategies=[s])
        risks = StrategyRiskDetector.detect(result)
        self.assertTrue(any("Complexity" in r.name for r in risks))

    def test_confidence_detector(self):
        m = ConfidenceMetric(name='test', score=0.4, weight=0.5, reason='')
        a = ConfidenceAnalysis(component='forecast', confidence_score=0.4, classification='Low',
                               metrics=[m], reasoning='', warnings=[])
        result = ConfidenceResult(success=True, overall_confidence=0.4, analyses=[a])
        risks = ConfidenceRiskDetector.detect(result)
        self.assertTrue(any("Low" in r.name for r in risks))

    def test_formatter(self):
        f = RiskFactor(id='1', name='Test', category=RiskCategory.GENERAL,
                       severity=0.5, probability=0.5, impact=0.5,
                       risk_score=RiskScorer.compute_risk_score(0.5, 0.5, 0.5),
                       reason='r', mitigation='m', source='test')
        a = RiskAnalysis(component='forecast', overall_risk=0.5, classification='Medium',
                         risk_factors=[f], warnings=[], summary='s')
        result = RiskResult(success=True, overall_risk=0.5, analyses=[a])
        text = RiskFormatter.to_text(result)
        self.assertIn('Risk Report', text)
        markdown = RiskFormatter.to_markdown(result)
        self.assertIn('# Risk Report', markdown)
        d = RiskFormatter.to_dict(result)
        self.assertEqual(d['overall_risk'], 0.5)
        json_output = RiskFormatter.to_json(result)
        self.assertIn('0.5', json_output)

    def test_component_aggregation_policy_defined(self):
        self.assertIn(COMPONENT_RISK_AGGREGATION, ['max', 'weighted_average', 'top3_average'])

if __name__ == '__main__':
    unittest.main()
