import unittest
import os

class TestImportSideEffects(unittest.TestCase):
    def test_import_forecast_ai_does_not_trigger_simulator_or_models(self):
        try:
            import core.forecast_ai
        except ImportError:
            self.skipTest("ForecastAI not properly installed in PYTHONPATH")
        self.assertFalse(hasattr(core.forecast_ai, 'simulator'))
        self.assertFalse(hasattr(core.forecast_ai, 'oh_predictor'))
        self.assertFalse(hasattr(core.forecast_ai, 'nps_predictor'))
        cwd_files = os.listdir('.')
        self.assertNotIn('simulator.log', cwd_files)
