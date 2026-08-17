import unittest
from core.forecast_ai.base_engine import ForecastAIEngine
from core.forecast_ai.models import ForecastRequest
from core.forecast_ai.engines import (
    ForecastOrchestrator, ReverseOptimizer,
    RecommendationEngine, StrategyEngine
)

class TestEngineContract(unittest.TestCase):
    def test_all_engines_have_execute_and_return_response(self):
        engines = [
            ForecastOrchestrator(), ReverseOptimizer(),
            RecommendationEngine(), StrategyEngine()
        ]
        req = ForecastRequest(operation="forecast")
        for eng in engines:
            with self.subTest(engine=eng.__class__.__name__):
                self.assertTrue(hasattr(eng, "execute"))
                resp = eng.execute(req)
                from core.forecast_ai.models import ForecastResponse
                self.assertIsInstance(resp, ForecastResponse)
                self.assertTrue(hasattr(resp, "success"))
                self.assertTrue(hasattr(resp, "timestamp"))
                self.assertNotEqual(resp.timestamp, "")
