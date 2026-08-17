import importlib


def test_test_training_has_behavioral_tests():
    """The NPS training suite must contain real behavioral tests (not the old
    assertTrue(True) placeholder)."""
    module = importlib.import_module("core.nps_predictor.tests.test_training")
    assert hasattr(module, "test_training_lifecycle_and_persistence")
    assert hasattr(module, "test_selection_sampling_respects_row_limit")
