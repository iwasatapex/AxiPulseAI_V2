import unittest
from core.forecast_ai.reporting import (
    ReportEngine, ReportResult, ReportType,
    ReportBuilder, ReportExporter, ReportTemplates,
    SectionGenerator
)
from core.forecast_ai.trends import TrendAnalysis, TrendResult
from core.forecast_ai.sensitivity import SensitivityAnalysis, SensitivityResult
from core.forecast_ai.recommendations import Recommendation, Category, Priority, Difficulty, RecommendationResult
from core.forecast_ai.strategy import StrategyPlan, StrategyCategory, StrategyResult
from core.forecast_ai.confidence import ConfidenceResult, ConfidenceAnalysis, ConfidenceMetric
from core.forecast_ai.risk import RiskResult, RiskAnalysis, RiskFactor, RiskCategory

class TestReportRefinements(unittest.TestCase):
    def setUp(self):
        self.forecast = {"timeline": [{"operations_health": 80, "nps": 70},
                                      {"operations_health": 78, "nps": 68},
                                      {"operations_health": 75, "nps": 65}]}
        self.trend = TrendResult(success=True, analyses=[TrendAnalysis(metric='OH', trend_direction='Decrease',
                              trend_strength='Strong', moving_average=[], minimum=0, maximum=0,
                              mean=0, median=0, variance=0, standard_deviation=0, volatility='High',
                              absolute_change=0, percent_change=0, pattern='Decreasing', confidence=0)])
        self.sens = SensitivityResult(success=True, analyses=[SensitivityAnalysis(metric='quality',
                           baseline_output_oh=80, baseline_output_nps=70,
                           modified_output_oh=90, modified_output_nps=75,
                           operations_health_change=10, nps_change=5,
                           sensitivity_score_oh=2.0, sensitivity_score_nps=1.0,
                           elasticity_oh=0.5, elasticity_nps=0.4, rank=0, classification='High')])
        self.recs = RecommendationResult(success=True, recommendations=[
            Recommendation(id='1', title='Improve Quality', description='', category=Category.QUALITY,
                           priority=Priority.HIGH, difficulty=Difficulty.MEDIUM,
                           estimated_operations_health_gain=3.0)
        ])
        self.strategy = StrategyResult(success=True, strategies=[
            StrategyPlan(id='1', name='Quality Strategy', description='', objective='',
                         category=StrategyCategory.GENERAL, priority='High', estimated_duration_weeks=4,
                         estimated_complexity=0.5, estimated_disruption=0.3, confidence=0.7,
                         recommendations=[], timeline=[], milestones=[], risks=[])
        ])
        self.confidence = ConfidenceResult(success=True, overall_confidence=0.85,
                                           analyses=[ConfidenceAnalysis(component='forecast',
                                             confidence_score=0.85, classification='High',
                                             metrics=[], reasoning='', warnings=[])])
        self.risk = RiskResult(success=True, overall_risk=0.3,
                               analyses=[RiskAnalysis(component='forecast', overall_risk=0.3,
                                        classification='Low', risk_factors=[], warnings=[], summary='')])

    def test_executive_summary_synthesis(self):
        result = ReportBuilder.build(
            forecast_result=self.forecast,
            trend_result=self.trend,
            sensitivity_result=self.sens,
            recommendation_result=self.recs,
            strategy_result=self.strategy,
            confidence_result=self.confidence,
            risk_result=self.risk
        )
        self.assertIsNotNone(result.executive_summary)
        self.assertGreater(len(result.executive_summary.key_findings), 0)
        self.assertGreater(len(result.executive_summary.top_recommendations), 0)
        self.assertIn('Favorable', result.executive_summary.operational_outlook)

    def test_forecast_section_rich(self):
        sec = SectionGenerator.forecast_section(self.forecast, self.confidence)
        self.assertIn('Horizon: 3 days', sec.content)
        self.assertIn('Average OH: 77.7', sec.content)
        self.assertIn('Net OH change: -5.0', sec.content)
        self.assertIn('Confidence: 85%', sec.content)

    def test_appendix_generation(self):
        result = ReportBuilder.build(forecast_result=self.forecast)
        self.assertIsNotNone(result.appendix)
        self.assertIn('Components', result.appendix.components)

    def test_template_ordering(self):
        template = ReportTemplates.get_template('technical')
        self.assertIn('sensitivity', template['sections'])
        template_exec = ReportTemplates.get_template('executive')
        self.assertNotIn('sensitivity', template_exec['sections'])

    def test_exporter_refactored(self):
        result = ReportBuilder.build(forecast_result=self.forecast,
                                     recommendation_result=self.recs,
                                     confidence_result=self.confidence)
        text = ReportExporter.to_text(result)
        self.assertIn('Executive Summary', text)
        self.assertIn('Detailed Sections', text)
        markdown = ReportExporter.to_markdown(result)
        self.assertIn('Executive Summary', markdown)
        self.assertIn('Detailed Sections', markdown)

    def test_different_report_types(self):
        engine = ReportEngine()
        result_exec = engine.generate(report_type='executive', forecast_result=self.forecast)
        result_tech = engine.generate(report_type='technical', forecast_result=self.forecast)
        self.assertNotEqual(result_exec.title, result_tech.title)

if __name__ == '__main__':
    unittest.main()
