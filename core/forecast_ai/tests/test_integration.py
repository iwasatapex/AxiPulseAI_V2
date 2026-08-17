import unittest
from core.forecast_ai.planner import ForecastAIPlanner
from core.forecast_ai.models import ForecastRequest
from core.forecast_ai.prediction.provider import PredictorProvider

class TestIntegration(unittest.TestCase):
    def test_end_to_end(self):
        PredictorProvider.reset()
        state = {"quality": 85, "competency": 78, "release": 92, "transfer": 12, "attendance": 90}
        req = ForecastRequest(operation="forecast", horizon=1, parameters={"state": state})
        resp = ForecastAIPlanner().execute(req)
        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "ForecastOrchestrator")
        self.assertIsNotNone(resp.payload)
        day = resp.payload["timeline"][0]
        self.assertIsInstance(day["operations_health"], (int, float))
        self.assertIsInstance(day["nps"], (int, float))
