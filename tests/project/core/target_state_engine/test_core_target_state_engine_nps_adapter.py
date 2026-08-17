import importlib

def test_nps_adapter_surface():
    module = importlib.import_module("core.target_state_engine.nps_adapter")
    assert hasattr(module, "convert_output")
