import importlib

def test_engine_surface():
    module = importlib.import_module("core.target_state_engine.engine")
    assert hasattr(module, "find_target_state")
    assert hasattr(module, "TargetStateEngine")
