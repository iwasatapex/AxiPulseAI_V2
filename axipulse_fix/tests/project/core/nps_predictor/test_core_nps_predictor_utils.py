import importlib

def test_utils_surface():
    module = importlib.import_module("core.nps_predictor.utils")
    assert hasattr(module, "safe_divide")
    assert hasattr(module, "round_to_int")
    assert hasattr(module, "ensure_datetime")
