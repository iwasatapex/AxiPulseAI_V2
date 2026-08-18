import importlib

def test_council_surface():
    module = importlib.import_module("core.target_state_engine.council")
    assert hasattr(module, "predict")
    assert hasattr(module, "analyze")
    assert hasattr(module, "weighted_consensus")
    assert hasattr(module, "summarize")
    assert hasattr(module, "ModelCouncil")
