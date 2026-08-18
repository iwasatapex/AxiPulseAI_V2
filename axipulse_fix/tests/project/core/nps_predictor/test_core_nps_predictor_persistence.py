import importlib

def test_persistence_surface():
    module = importlib.import_module("core.nps_predictor.persistence")
    assert hasattr(module, "save_model")
    assert hasattr(module, "load_model")
