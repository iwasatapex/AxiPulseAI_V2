import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.state.models")
    assert hasattr(module, "to_dict")
    assert hasattr(module, "from_dict")
    assert hasattr(module, "OperationalState")
