import unittest
import threading
from core.forecast_ai.planner import ForecastAIPlanner
from core.forecast_ai.models import ForecastRequest

class TestThreadSafety(unittest.TestCase):
    def test_multiple_planners_independent(self):
        planners = [ForecastAIPlanner() for _ in range(3)]
        results = []
        def run(planner, idx):
            state = {"quality": 85.0, "competency": 78.0, "release": 92.0, "transfer": 12.0, "attendance": 90.0}
            req = ForecastRequest(operation="forecast", horizon=1, parameters={"state": state})
            resp = planner.execute(req)
            results.append((idx, resp.success, resp.engine))
        threads = []
        for i, pl in enumerate(planners):
            t = threading.Thread(target=run, args=(pl, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 3)
        for idx, success, engine in results:
            self.assertTrue(success)
            self.assertEqual(engine, "ForecastOrchestrator")
