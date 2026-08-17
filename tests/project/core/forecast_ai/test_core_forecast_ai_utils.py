import importlib

def test_utils_surface():
    module = importlib.import_module("core.forecast_ai.utils")
    assert hasattr(module, "validate_inputs")
    assert hasattr(module, "convert_to_serializable")
    assert hasattr(module, "safe_divide")
    assert hasattr(module, "clamp")
    assert hasattr(module, "date_range")
