import importlib

def test_engine_persistence_surface():
    module = importlib.import_module("api.services.engine_persistence")
    assert hasattr(module, "save_adie_decision")
    assert hasattr(module, "save_model_prediction")
    assert hasattr(module, "EnginePersistence")
