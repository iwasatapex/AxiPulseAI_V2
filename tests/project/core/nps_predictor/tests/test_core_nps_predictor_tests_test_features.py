import importlib


def test_test_features_has_behavioral_tests():
    """The NPS feature suite must contain real behavioral tests (not the old
    assertTrue(True) placeholder)."""
    module = importlib.import_module("core.nps_predictor.tests.test_features")
    assert hasattr(module, "test_align_features_exact_order")
    assert hasattr(module, "test_prepare_features_produces_finite_features")
