import unittest
from copy import deepcopy
from core.forecast_ai.optimization import (
    ReverseOptimizer,
    TargetGoal,
    Constraint,
    ConstraintType,
    OptimizationRequest,
    ScoreCalculator,
    ConstraintValidator
)
from core.forecast_ai.prediction.provider import PredictorProvider
from core.forecast_ai.state import OperationalState
from core.forecast_ai.scenarios import ScenarioManager

class DummyPredictor:
    def predict(self, state):
        quality = state.get("quality", state.get("actual_quality", 80.0))
        competency = state.get("competency", state.get("actual_competency", 70.0))

        oh = 80.0 + quality * 0.1
        nps = 70.0 + competency * 0.05

        return oh, nps

class TestOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PredictorProvider.set_oh_predictor(DummyPredictor())
        PredictorProvider.set_nps_predictor(DummyPredictor())

    def setUp(self):
        self.state = {"quality": 80, "competency": 70, "attendance": 90, "release": 85, "transfer": 10}
        self.optimizer = ReverseOptimizer()

    def test_oh_only_target(self):
        target = TargetGoal(target_operations_health=90.0, tolerance=1.0)
        req = OptimizationRequest(initial_state=self.state, target_goal=target, max_iterations=50)
        result = self.optimizer.optimize(req)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.best_solution.predicted_operations_health, 90.0, delta=1.0)

    def test_nps_only_target(self):
        target = TargetGoal(target_nps=75.0, tolerance=1.0)
        req = OptimizationRequest(initial_state=self.state, target_goal=target, max_iterations=50)
        result = self.optimizer.optimize(req)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.best_solution.predicted_nps, 75.0, delta=1.0)

    def test_constraint_fixed(self):
        constraints = [Constraint(field="attendance", type=ConstraintType.FIXED, value=90.0)]
        target = TargetGoal(target_operations_health=90.0, tolerance=1.0, constraints=constraints)
        req = OptimizationRequest(initial_state=self.state, target_goal=target, max_iterations=50)
        result = self.optimizer.optimize(req)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.best_solution.state["attendance"], 90.0, delta=0.001)

    def test_scenario_integration(self):
        # Ensure ScenarioManager is called (we can mock but just check no error)
        # Since we have no active scenarios, it's a pass-through.
        target = TargetGoal(target_operations_health=90.0, tolerance=1.0)
        req = OptimizationRequest(initial_state=self.state, target_goal=target, max_iterations=30)
        result = self.optimizer.optimize(req)
        self.assertTrue(result.success)

if __name__ == '__main__':
    unittest.main()
