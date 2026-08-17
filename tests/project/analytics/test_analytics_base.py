import importlib

def test_base_surface():
    module = importlib.import_module("analytics.base")
    assert hasattr(module, "load_data")
    assert hasattr(module, "to_json")
    assert hasattr(module, "AnalyticsBase")
