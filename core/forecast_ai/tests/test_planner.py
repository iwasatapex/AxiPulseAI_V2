import unittest
from core.forecast_ai.planner import ForecastAIPlanner
from core.forecast_ai.models import ForecastRequest

class TestPlannerEndToEnd(unittest.TestCase):
    def test_forecast_request_success(self):
        state = {
            "quality": 85.0,
            "competency": 78.0,
            "release": 92.0,
            "transfer": 12.0,
            "attendance": 90.0
        }
        req = ForecastRequest(
            operation="forecast",
            horizon=1,
            parameters={"state": state}
        )
        planner = ForecastAIPlanner()
        resp = planner.execute(req)
        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "ForecastOrchestrator")
        self.assertEqual(resp.operation, "forecast")
        self.assertIsNotNone(resp.payload)
        self.assertIn("timeline", resp.payload)
        self.assertEqual(len(resp.payload["timeline"]), 1)

    def test_forecast_without_state_fails(self):
        req = ForecastRequest(operation="forecast", horizon=1)
        planner = ForecastAIPlanner()
        resp = planner.execute(req)
        self.assertFalse(resp.success)
        self.assertIn("No operational state provided", resp.errors[0])

    def test_invalid_operation(self):
        req = ForecastRequest(operation="banana")
        planner = ForecastAIPlanner()
        resp = planner.execute(req)
        self.assertFalse(resp.success)
        self.assertIn("Unsupported operation: banana", resp.errors)

    def test_missing_operation(self):
        req = ForecastRequest(operation=None)  # type: ignore
        planner = ForecastAIPlanner()
        resp = planner.execute(req)
        self.assertFalse(resp.success)
        self.assertIn("Missing 'operation' field", resp.errors)

class TestRouting(unittest.TestCase):
    def setUp(self):
        self.planner = ForecastAIPlanner()
        self.operations = [
            ("forecast", "ForecastOrchestrator"),
            ("reverse_optimize", "ReverseOptimizer"),
            ("recommend", "RecommendationEngine"),
            ("strategy", "StrategyEngine"),
            # Other operations not yet implemented
        ]
    def test_router_returns_correct_engine(self):
        for op, expected_engine_name in self.operations:
            req = ForecastRequest(operation=op)
            engine = self.planner.route(req)
            self.assertIsNotNone(engine)
            self.assertEqual(engine.__class__.__name__, expected_engine_name)
