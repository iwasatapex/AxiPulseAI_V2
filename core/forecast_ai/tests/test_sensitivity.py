import unittest
from core.forecast_ai.sensitivity import (
    SensitivityEngine, SensitivityAnalysis, SensitivityResult,
    ExperimentGenerator, SensitivityAnalyzer, SensitivityRanker,
    SensitivityFormatter
)
from core.forecast_ai.prediction.provider import PredictorProvider
from core.forecast_ai.config import KPI_BOUNDS, SENSITIVITY_THRESHOLDS

class DummyPredictor:
    def predict(self, state):
        # Mock: OH = 80 + 0.1*quality + 0.05*competency, NPS = 70 + 0.08*quality + 0.04*attendance
        oh = 80.0 + 0.1 * state.get('quality', 80.0) + 0.05 * state.get('competency', 70.0)
        nps = 70.0 + 0.08 * state.get('quality', 80.0) + 0.04 * state.get('attendance', 90.0)
        return oh, nps

class TestSensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PredictorProvider.set_oh_predictor(DummyPredictor())
        PredictorProvider.set_nps_predictor(DummyPredictor())

    def setUp(self):
        self.state = {
            "quality": 85.0,
            "competency": 78.0,
            "attendance": 90.0,
            "release": 92.0,
            "transfer": 12.0
        }
        self.engine = SensitivityEngine(step_size=1.0)

    def test_symmetric_experiments(self):
        gen = ExperimentGenerator()
        experiments = gen.generate_experiments(self.state, step_size=1.0)
        # Should have both + and - for each metric that is within bounds
        # Quality, competency, attendance, release, transfer
        metrics = set(e['metric'] for e in experiments)
        self.assertEqual(metrics, {'quality', 'competency', 'attendance', 'release', 'transfer'})
        # Check that we have both directions
        plus = [e for e in experiments if e.get('direction') == '+']
        minus = [e for e in experiments if e.get('direction') == '-']
        self.assertGreater(len(plus), 0)
        self.assertGreater(len(minus), 0)

    def test_bounds_respected(self):
        gen = ExperimentGenerator()
        state = {"quality": 0.0, "competency": 100.0}  # at extremes
        experiments = gen.generate_experiments(state, step_size=1.0, metrics=['quality', 'competency'])
        # quality should only have + (since - would go below 0)
        # competency should only have - (since + would go above 100)
        quality_exps = [e for e in experiments if e['metric'] == 'quality']
        competency_exps = [e for e in experiments if e['metric'] == 'competency']
        self.assertEqual(len(quality_exps), 1)
        self.assertEqual(quality_exps[0]['direction'], '+')
        self.assertEqual(len(competency_exps), 1)
        self.assertEqual(competency_exps[0]['direction'], '-')

    def test_analyzer_aggregation(self):
        raw_results = [
            {'metric': 'quality', 'direction': '+', 'delta': 1.0,
             'oh_change': 0.8, 'nps_change': 0.6,
             'sensitivity_oh': 0.8, 'sensitivity_nps': 0.6,
             'elasticity_oh': 0.9, 'elasticity_nps': 0.7,
             'modified_oh': 81.0, 'modified_nps': 71.0},
            {'metric': 'quality', 'direction': '-', 'delta': -1.0,
             'oh_change': -0.7, 'nps_change': -0.5,
             'sensitivity_oh': 0.7, 'sensitivity_nps': 0.5,
             'elasticity_oh': 0.8, 'elasticity_nps': 0.6,
             'modified_oh': 79.0, 'modified_nps': 69.0}
        ]
        analyzer = SensitivityAnalyzer()
        agg = analyzer.aggregate(raw_results, 'quality', 80.0, 70.0)
        self.assertEqual(agg.metric, 'quality')
        self.assertAlmostEqual(agg.sensitivity_score_oh, 0.75, places=2)
        self.assertAlmostEqual(agg.sensitivity_score_nps, 0.55, places=2)
        self.assertGreater(agg.confidence, 0.5)

    def test_engine_full(self):
        result = self.engine.analyze(self.state)
        self.assertTrue(result.success)
        self.assertEqual(len(result.analyses), 5)
        self.assertEqual(len(result.ranking), 5)
        # Check ranking: sorted by absolute OH sensitivity
        sens = [a.sensitivity_score_oh for a in result.ranking]
        # Quality has coefficient 0.1, competency 0.05, attendance 0.0 (NPS only), release 0.0, transfer 0.0
        # So ranking should be quality, competency, then the rest
        self.assertEqual(result.ranking[0].metric, 'quality')

    def test_formatter(self):
        analysis = SensitivityAnalysis(
            metric='quality',
            baseline_output_oh=80.0,
            baseline_output_nps=70.0,
            modified_output_oh=81.0,
            modified_output_nps=71.0,
            operations_health_change=1.0,
            nps_change=1.0,
            sensitivity_score_oh=1.0,
            sensitivity_score_nps=1.0,
            elasticity_oh=0.5,
            elasticity_nps=0.4,
            classification='High',
            rank=1,
            confidence=0.9
        )
        text = SensitivityFormatter.to_text([analysis])
        self.assertIn('quality', text)
        markdown = SensitivityFormatter.to_markdown([analysis])
        self.assertIn('## quality', markdown)
        as_dict = SensitivityFormatter.to_dict([analysis])
        self.assertEqual(as_dict[0]['metric'], 'quality')
        json_output = SensitivityFormatter.to_json([analysis])
        self.assertIn('quality', json_output)

if __name__ == '__main__':
    unittest.main()
