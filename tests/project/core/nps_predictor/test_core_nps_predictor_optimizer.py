import importlib


def test_optimizer_surface():
    module = importlib.import_module("core.nps_predictor.optimizer")

    assert hasattr(module, "reverse_optimize_nps")
