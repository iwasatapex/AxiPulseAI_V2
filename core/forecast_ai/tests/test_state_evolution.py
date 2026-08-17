import unittest
from core.forecast_ai.state import OperationalState, StateEvolutionEngine
from core.forecast_ai.models import PredictionResult

class TestStateEvolution(unittest.TestCase):
    def setUp(self):
        self.engine = StateEvolutionEngine()
        self.initial = OperationalState(
            quality=85.0,
            competency=78.0,
            transfer=12.0,
            release=92.0,
            attendance=90.0,
            operations_health=75.0,
            nps=65.0
        )

    def test_evolve_returns_new_object(self):
        pred = PredictionResult(operations_health=82.0, nps=70.0, warnings=[], errors=[])
        new_state = self.engine.evolve(self.initial, pred)
        self.assertIsNot(new_state, self.initial)

    def test_original_unchanged(self):
        pred = PredictionResult(operations_health=82.0, nps=70.0, warnings=[], errors=[])
        _ = self.engine.evolve(self.initial, pred)
        self.assertEqual(self.initial.quality, 85.0)
        self.assertEqual(self.initial.operations_health, 75.0)

    def test_oh_nps_propagated(self):
        pred = PredictionResult(operations_health=82.0, nps=70.0, warnings=[], errors=[])
        new_state = self.engine.evolve(self.initial, pred)
        self.assertEqual(new_state.operations_health, 82.0)
        self.assertEqual(new_state.nps, 70.0)

    def test_metadata_preserved(self):
        self.initial.metadata = {"source": "test"}
        pred = PredictionResult(operations_health=82.0, nps=70.0, warnings=[], errors=[])
        new_state = self.engine.evolve(self.initial, pred)
        self.assertEqual(new_state.metadata, {"source": "test"})

    def test_none_predictions_keep_old_values(self):
        pred = PredictionResult(operations_health=None, nps=None, warnings=[], errors=["Failed"])
        new_state = self.engine.evolve(self.initial, pred)
        self.assertEqual(new_state.operations_health, self.initial.operations_health)
        self.assertEqual(new_state.nps, self.initial.nps)

    def test_multiple_evolutions(self):
        pred1 = PredictionResult(operations_health=82.0, nps=70.0, warnings=[], errors=[])
        state1 = self.engine.evolve(self.initial, pred1)
        pred2 = PredictionResult(operations_health=85.0, nps=72.0, warnings=[], errors=[])
        state2 = self.engine.evolve(state1, pred2)
        self.assertEqual(state2.operations_health, 85.0)
        self.assertEqual(state2.nps, 72.0)

if __name__ == '__main__':
    unittest.main()
