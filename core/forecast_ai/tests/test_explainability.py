import unittest
from core.forecast_ai.explainability import (
    ExplainabilityEngine, ExplainabilityResult, Explanation, Evidence,
    ExplainabilityAnalyzer, ReasoningBuilder, TraceBuilder,
    ExplanationTemplates, ExplainabilityFormatter
)
from core.forecast_ai.trends import TrendAnalysis, TrendResult
from core.forecast_ai.sensitivity import SensitivityAnalysis, SensitivityResult
from core.forecast_ai.recommendations import Recommendation, Category, Priority, Difficulty, RecommendationResult
from core.forecast_ai.strategy import StrategyPlan, StrategyCategory, StrategyResult
from core.forecast_ai.confidence import ConfidenceResult, ConfidenceAnalysis, ConfidenceMetric
from core.forecast_ai.risk import RiskResult, RiskAnalysis, RiskFactor, RiskCategory

class TestExplainabilityImprovements(unittest.TestCase):
    def test_inherited_confidence(self):
        # Create confidence result
        c = ConfidenceResult(success=True, overall_confidence=0.85, analyses=[])
        forecast = {"timeline": [{"operations_health": 80}, {"operations_health": 82}]}
        expl = ExplainabilityAnalyzer.analyze_forecast(forecast, confidence_result=c)
        self.assertEqual(expl.confidence, 0.85)

    def test_forecast_uses_cross_component_data(self):
        # Create trend and sensitivity results
        trend = TrendResult(success=True, analyses=[TrendAnalysis(metric='quality', trend_direction='Decrease',
                                                                 trend_strength='Strong', moving_average=[], minimum=0,
                                                                 maximum=0, mean=0, median=0, variance=0,
                                                                 standard_deviation=0, volatility='High',
                                                                 absolute_change=0, percent_change=0,
                                                                 pattern='Decreasing', confidence=0)])
        sens = SensitivityResult(success=True, analyses=[SensitivityAnalysis(metric='quality',
                           baseline_output_oh=80, baseline_output_nps=70,
                           modified_output_oh=90, modified_output_nps=75,
                           operations_health_change=10, nps_change=5,
                           sensitivity_score_oh=2.0, sensitivity_score_nps=1.0,
                           elasticity_oh=0.5, elasticity_nps=0.4, rank=0,
                           classification='High')], ranking=[])
        forecast = {"timeline": [{"operations_health": 80}, {"operations_health": 75}]}
        expl = ExplainabilityAnalyzer.analyze_forecast(forecast, trend_result=trend, sensitivity_result=sens)
        # Should have evidence from trend and sensitivity
        evidence_fields = [e.field for e in expl.evidence]
        self.assertIn('quality', evidence_fields)  # from trend
        self.assertIn('quality', evidence_fields)  # from sensitivity
        self.assertIn('forecast', expl.reasoning)
        self.assertGreater(len(expl.source_chain), 2)  # includes trend and sensitivity

    def test_trace_builder_includes_dependencies(self):
        traces = TraceBuilder.build_trace(['forecast', 'trend'])
        # Should have PredictionService, ForecastOrchestrator, TrendEngine, ExplainabilityEngine
        engine_names = [t.engine for t in traces]
        self.assertIn('PredictionService', engine_names)
        self.assertIn('ForecastOrchestrator', engine_names)
        self.assertIn('TrendEngine', engine_names)
        self.assertIn('ExplainabilityEngine', engine_names)
        # Check dependencies
        for t in traces:
            if t.engine == 'ForecastOrchestrator':
                self.assertIn('PredictionService', t.dependencies)

    def test_evidence_reference(self):
        ev = Evidence(component='forecast', field='oh', value=80, importance='High',
                      description='OH', reference='forecast.timeline[0].operations_health')
        self.assertEqual(ev.reference, 'forecast.timeline[0].operations_health')

    def test_reasoning_builder_narrative(self):
        evidence = [Evidence(component='test', field='test', value=1, importance='High',
                             description='test evidence', reference='test')]
        template = {'reasoning_template': 'The forecast shows {direction} because {evidence}.'}
        metadata = {'direction': 'decrease'}
        reason = ReasoningBuilder.build_reasoning('forecast', evidence, template, metadata)
        self.assertIn('decrease', reason)
        self.assertIn('test evidence', reason)

    def test_template_expansion(self):
        template = ExplanationTemplates.get_template('forecast')
        self.assertIn('summary_template', template)
        self.assertIn('reasoning_template', template)
        self.assertIn('conclusion_template', template)

    def test_deterministic_explanation_id(self):
        forecast = {"timeline": [{"operations_health": 80}]}
        expl1 = ExplainabilityAnalyzer.analyze_forecast(forecast)
        expl2 = ExplainabilityAnalyzer.analyze_forecast(forecast)
        self.assertEqual(expl1.id, expl2.id)

if __name__ == '__main__':
    unittest.main()
