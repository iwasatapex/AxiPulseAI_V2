import importlib

def test_intelligence_surface():
    module = importlib.import_module("core.decision_intelligence.v3.intelligence")
    assert hasattr(module, "analyze")
    assert hasattr(module, "ProbabilisticDecision")
    assert hasattr(module, "ADIEProbabilisticEngine")
