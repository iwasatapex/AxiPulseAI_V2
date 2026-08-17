import importlib

def test_persistence_service_surface():
    module = importlib.import_module("api.services.persistence_service")
    assert hasattr(module, "save_decision")
    assert hasattr(module, "save_prediction")
    assert hasattr(module, "PersistenceService")
