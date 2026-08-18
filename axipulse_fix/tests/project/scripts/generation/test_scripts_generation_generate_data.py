import importlib

def test_generate_data_surface():
    module = importlib.import_module("scripts.generation.generate_data")
    assert hasattr(module, "clamp")
    assert hasattr(module, "get_season")
    assert hasattr(module, "week_position_modifier")
    assert hasattr(module, "sample_complexity")
    assert hasattr(module, "call_volume")
    assert hasattr(module, "is_exceptional")
    assert hasattr(module, "generate_intelligence_factor")
    assert hasattr(module, "generate_weekdays")
    assert hasattr(module, "simulate")
    assert hasattr(module, "main")
