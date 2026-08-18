"""
Unit tests for AxiPulseAI
"""
import unittest

class TestNPSPredictor(unittest.TestCase):
    def test_import(self):
        from core.nps_predictor import NPSPredictor
        self.assertIsNotNone(NPSPredictor)
