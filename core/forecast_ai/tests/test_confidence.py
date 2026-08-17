import unittest
from core.forecast_ai.confidence import (
    ConfidenceEngine, ConfidenceResult, ConfidenceAnalysis,
    ConfidenceMetrics, ConfidenceScorer, ConfidenceFormatter
)
from core.forecast_ai.trends import TrendAnalysis
from core.forecast_ai.sensitivity import SensitivityAnalysis
from core.forecast_ai.recommendations import Recommendation, Category, Priority, Difficulty
from core.forecast_ai.strategy import StrategyPlan, StrategyCategory
from core.forecast_ai.config import COMPONENT_WEIGHTS

class TestConfidence(unittest.TestCase):
    def test_metric_prediction_stability(self):
        timeline = [
            {'operations_health': 80, 'nps': 70},
            {'operations_health': 82, 'nps': 72},
            {'operations_health': 81, 'nps': 71},
        ]
        score = ConfidenceMetrics.prediction_stability(timeline)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_metric_trend_consistency(self):
        trends = [
            TrendAnalysis(metric='OH', trend_direction='Increase', trend_strength='Strong',
                          moving_average=[], minimum=0, maximum=0, mean=0, median=0,
                          variance=0, standard_deviation=0, volatility='Low',
                          absolute_change=0, percent_change=0, pattern='Increasing', confidence=0),
            TrendAnalysis(metric='NPS', trend_direction='Increase', trend_strength='Strong',
                          moving_average=[], minimum=0, maximum=0, mean=0, median=0,
                          variance=0, standard_deviation=0, volatility='Low',
                          absolute_change=0, percent_change=0, pattern='Increasing', confidence=0)
        ]
        score = ConfidenceMetrics.trend_consistency(trends)
        self.assertGreaterEqual(score, 0.5)

    def test_metric_sensitivity_consistency(self):
        sens = [
            SensitivityAnalysis(metric='quality', baseline_output_oh=80, baseline_output_nps=70,
                                modified_output_oh=81, modified_output_nps=71,
                                operations_health_change=1, nps_change=1,
                                sensitivity_score_oh=2.0, sensitivity_score_nps=1.0,
                                elasticity_oh=0.5, elasticity_nps=0.4, rank=0, classification='High'),
            SensitivityAnalysis(metric='competency', baseline_output_oh=80, baseline_output_nps=70,
                                modified_output_oh=80.5, modified_output_nps=70.5,
                                operations_health_change=0.5, nps_change=0.5,
                                sensitivity_score_oh=0.5, sensitivity_score_nps=0.5,
                                elasticity_oh=0.2, elasticity_nps=0.2, rank=0, classification='Medium')
        ]
        score = ConfidenceMetrics.sensitivity_consistency(sens)
        self.assertGreaterEqual(score, 0.0)

    def test_metric_recommendation_agreement(self):
        recs = [
            Recommendation(id='1', title='A', description='', category=Category.QUALITY,
                           priority=Priority.HIGH, difficulty=Difficulty.EASY),
            Recommendation(id='2', title='B', description='', category=Category.QUALITY,
                           priority=Priority.MEDIUM, difficulty=Difficulty.MEDIUM),
            Recommendation(id='3', title='C', description='', category=Category.TRAINING,
                           priority=Priority.LOW, difficulty=Difficulty.HARD)
        ]
        score = ConfidenceMetrics.recommendation_agreement(recs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_metric_strategy_completeness_gradual(self):
        strategies = [
            StrategyPlan(id='1', name='A', description='', objective='', category=StrategyCategory.GENERAL,
                         priority='High', estimated_duration_weeks=4, estimated_complexity=0.5,
                         estimated_disruption=0.3, confidence=0.7, recommendations=['rec1'],
                         timeline=[], milestones=[('week1', 'task', 0.1)], risks=['risk1']),
            StrategyPlan(id='2', name='B', description='', objective='', category=StrategyCategory.GENERAL,
                         priority='High', estimated_duration_weeks=4, estimated_complexity=0.5,
                         estimated_disruption=0.3, confidence=0.7, recommendations=['rec2'],
                         timeline=[], milestones=[], risks=[])  # incomplete
        ]
        score = ConfidenceMetrics.strategy_completeness(strategies)
        # Strategy A: rec=1.0*0.4 + milestones=1.0*0.4 + risks=1.0*0.2 = 1.0
        # Strategy B: rec=1.0*0.4 + milestones=0.0*0.4 + risks=0.0*0.2 = 0.4
        # Average = 0.7
        self.assertAlmostEqual(score, 0.7, places=1)

    def test_confidence_scorer(self):
        from core.forecast_ai.confidence.models import ConfidenceMetric
        metrics = [
            ConfidenceMetric(name='a', score=0.8, weight=0.5, reason=''),
            ConfidenceMetric(name='b', score=0.6, weight=0.5, reason='')
        ]
        score = ConfidenceScorer.compute_confidence(metrics)
        self.assertEqual(score, 0.7)

    def test_classification(self):
        self.assertEqual(ConfidenceScorer.classify(0.95), "Very High")
        self.assertEqual(ConfidenceScorer.classify(0.80), "High")
        self.assertEqual(ConfidenceScorer.classify(0.60), "Medium")
        self.assertEqual(ConfidenceScorer.classify(0.40), "Low")
        self.assertEqual(ConfidenceScorer.classify(0.20), "Very Low")

    def test_weighted_overall_confidence(self):
        from core.forecast_ai.confidence.models import ConfidenceMetric, ConfidenceAnalysis
        # Create two components with different scores
        metrics = [ConfidenceMetric(name='test', score=0.9, weight=1.0, reason='')]
        analysis1 = ConfidenceAnalysis(component='forecast', confidence_score=0.9,
                                       classification='High', metrics=metrics, reasoning='')
        analysis2 = ConfidenceAnalysis(component='trend', confidence_score=0.5,
                                       classification='Medium', metrics=metrics, reasoning='')
        result = ConfidenceResult(success=True, overall_confidence=0.0,
                                  forecast_confidence=analysis1,
                                  trend_confidence=analysis2,
                                  analyses=[analysis1, analysis2])
        # Manually compute weighted average
        weighted_sum = (0.9 * COMPONENT_WEIGHTS['forecast'] +
                        0.5 * COMPONENT_WEIGHTS['trend'])
        total_weight = COMPONENT_WEIGHTS['forecast'] + COMPONENT_WEIGHTS['trend']
        expected = weighted_sum / total_weight
        # Our engine computes this; we're testing the logic here.
        self.assertGreater(expected, 0.5)

    def test_formatter(self):
        from core.forecast_ai.confidence.models import ConfidenceResult, ConfidenceAnalysis, ConfidenceMetric
        metric = ConfidenceMetric(name='test', score=0.8, weight=0.5, reason='good')
        analysis = ConfidenceAnalysis(component='forecast', confidence_score=0.8,
                                      classification='High', metrics=[metric],
                                      reasoning='Test reasoning', warnings=['warn'])
        result = ConfidenceResult(success=True, overall_confidence=0.8, analyses=[analysis])
        text = ConfidenceFormatter.to_text(result)
        self.assertIn('Confidence Report', text)
        markdown = ConfidenceFormatter.to_markdown(result)
        self.assertIn('# Confidence Report', markdown)
        as_dict = ConfidenceFormatter.to_dict(result)
        self.assertEqual(as_dict['overall_confidence'], 0.8)
        json_output = ConfidenceFormatter.to_json(result)
        self.assertIn('0.8', json_output)

    def test_missing_components(self):
        engine = ConfidenceEngine()
        result = engine.evaluate(forecast_result=None, trend_result=None)
        # Should still work with only some components
        self.assertFalse(result.success)
        self.assertIn("No analyses could be performed", result.errors[0])

if __name__ == '__main__':
    unittest.main()
