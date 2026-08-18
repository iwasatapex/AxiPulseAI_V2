import unittest

from core.operation_health_predictor import OperationalHealthPredictor


class TestOpsHealth(unittest.TestCase):
    def test_predictor_initializes_untrained(self):
        predictor = OperationalHealthPredictor()
        # A fresh predictor is not trained until train() runs.
        self.assertIsNotNone(predictor)
        self.assertEqual(predictor.trained, False)
        self.assertEqual(list(predictor.feature_names or []), [])

    def test_predictor_surface(self):
        predictor = OperationalHealthPredictor()
        self.assertTrue(hasattr(predictor, "train"))
        self.assertTrue(hasattr(predictor, "predict"))
        self.assertTrue(hasattr(predictor, "save_model"))
        self.assertTrue(hasattr(predictor, "load_model"))


if __name__ == "__main__":
    unittest.main()
