import unittest
from unittest.mock import patch
from core.forecast_ai.prediction.provider import PredictorProvider
from core.forecast_ai.prediction.service import PredictionService
from core.forecast_ai.models import PredictionRequest, PredictionResult

class DummyPredictor:
    def predict(self, state):
        return 75.0

class TestPredictorProvider(unittest.TestCase):
    def setUp(self):
        PredictorProvider.reset()

    def test_provider_returns_predictor_instances(self):
        oh = PredictorProvider.get_oh_predictor()
        nps = PredictorProvider.get_nps_predictor()

        self.assertIsNotNone(oh)
        self.assertIsNotNone(nps)

    def test_provider_injection_works(self):
        dummy = DummyPredictor()
        PredictorProvider.set_oh_predictor(dummy)
        PredictorProvider.set_nps_predictor(dummy)
        self.assertIs(PredictorProvider.get_oh_predictor(), dummy)
        self.assertIs(PredictorProvider.get_nps_predictor(), dummy)

    def test_provider_reset(self):
        dummy = DummyPredictor()
        PredictorProvider.set_oh_predictor(dummy)
        PredictorProvider.reset()
        self.assertIsNotNone(PredictorProvider.get_oh_predictor())

class TestPredictionService(unittest.TestCase):
    def setUp(self):
        self.dummy_oh = DummyPredictor()
        self.dummy_nps = DummyPredictor()
        self.service = PredictionService(
            oh_predictor=self.dummy_oh,
            nps_predictor=self.dummy_nps
        )
        self.state = {"quality": 85, "competency": 78}
        self.request = PredictionRequest(state=self.state)

    def test_service_calls_both_predictors(self):
        with patch.object(self.dummy_oh, 'predict', return_value=88.0) as mock_oh, \
             patch.object(self.dummy_nps, 'predict', return_value=72.0) as mock_nps:
            result = self.service.predict(self.request)
            mock_oh.assert_called_once_with(self.state)
            mock_nps.assert_called_once_with(self.state)
            self.assertEqual(result.operations_health, 88.0)
            self.assertEqual(result.nps, 72.0)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.warnings, [])
            self.assertFalse(hasattr(result, 'timestamp'))

    def test_service_handles_predictor_failure_returns_none(self):
        def failing_predict(state):
            raise ValueError("OH failed")
        self.dummy_oh.predict = failing_predict
        result = self.service.predict(self.request)
        self.assertIsNone(result.operations_health)
        self.assertEqual(result.nps, 75.0)
        self.assertIn("OH prediction error", result.errors[0])

    def test_both_fail_returns_none_none(self):
        def failing_predict(state):
            raise ValueError("fail")
        self.dummy_oh.predict = failing_predict
        self.dummy_nps.predict = failing_predict
        result = self.service.predict(self.request)
        self.assertIsNone(result.operations_health)
        self.assertIsNone(result.nps)
        self.assertEqual(len(result.errors), 2)

    def test_predict_oh_only(self):
        with patch.object(self.dummy_oh, 'predict', return_value=95.0) as mock:
            result = self.service.predict_oh(self.state)
            mock.assert_called_once_with(self.state)
            self.assertEqual(result, 95.0)

    def test_predict_nps_only(self):
        with patch.object(self.dummy_nps, 'predict', return_value=68.0) as mock:
            result = self.service.predict_nps(self.state)
            mock.assert_called_once_with(self.state)
            self.assertEqual(result, 68.0)

class TestPredictionResultOptional(unittest.TestCase):
    def test_optional_fields(self):
        result = PredictionResult(operations_health=None, nps=None, warnings=[], errors=["OH failed"])
        self.assertIsNone(result.operations_health)
        self.assertIsNone(result.nps)
        self.assertEqual(result.errors, ["OH failed"])

if __name__ == '__main__':
    unittest.main()
