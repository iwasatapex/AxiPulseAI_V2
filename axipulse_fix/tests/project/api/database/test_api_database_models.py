import importlib

def test_models_surface():
    module = importlib.import_module("api.database.models")
    assert hasattr(module, "DecisionHistory")
    assert hasattr(module, "PredictionHistory")
