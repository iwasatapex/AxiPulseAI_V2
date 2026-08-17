import importlib

def test_decision_engine_surface():
    module = importlib.import_module("archive.legacy_decision.decision_engine.decision_engine")
    assert hasattr(module, "analyze")
    assert hasattr(module, "DecisionEngine")
