import unittest
from datetime import datetime, timedelta
from core.forecast_ai.planner import ForecastAIPlanner
from core.forecast_ai.models import ForecastRequest
from core.forecast_ai.prediction.provider import PredictorProvider

class TestRecursiveForecast(unittest.TestCase):
    def setUp(self):
        try:
            PredictorProvider.get_oh_predictor()
        except NotImplementedError:
            self.skipTest("PredictorProvider not implemented, skipping recursive tests")
        self.planner = ForecastAIPlanner()
        self.state = {"quality": 85, "competency": 78, "release": 92, "transfer": 12, "attendance": 90}

    def test_horizon_1(self):
        req = ForecastRequest(operation="forecast", horizon=1, parameters={"state": self.state})
        resp = self.planner.execute(req)
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.payload["timeline"]), 1)

    def test_horizon_7(self):
        req = ForecastRequest(operation="forecast", horizon=7, parameters={"state": self.state})
        resp = self.planner.execute(req)
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.payload["timeline"]), 7)

    def test_horizon_30(self):
        req = ForecastRequest(operation="forecast", horizon=30, parameters={"state": self.state})
        resp = self.planner.execute(req)
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.payload["timeline"]), 30)

    def test_horizon_365(self):
        req = ForecastRequest(operation="forecast", horizon=365, parameters={"state": self.state})
        resp = self.planner.execute(req)
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.payload["timeline"]), 365)

    def test_dates_increase(self):
        req = ForecastRequest(operation="forecast", horizon=5, parameters={"state": self.state})
        resp = self.planner.execute(req)
        dates = [day["date"] for day in resp.payload["timeline"]]
        dt_dates = [datetime.fromisoformat(d) for d in dates]
        for i in range(1, len(dt_dates)):
            self.assertEqual((dt_dates[i] - dt_dates[i-1]).days, 1)

if __name__ == '__main__':
    unittest.main()
