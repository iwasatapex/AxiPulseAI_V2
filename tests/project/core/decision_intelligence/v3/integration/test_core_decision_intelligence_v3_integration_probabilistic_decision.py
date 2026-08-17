import importlib

def test_probabilistic_decision_surface():
    module = importlib.import_module("core.decision_intelligence.v3.integration.probabilistic_decision")
    assert hasattr(module, "analyze")
    assert hasattr(module, "to_dict")
    assert hasattr(module, "ProbabilisticDecisionPackage")
    assert hasattr(module, "ProbabilisticDecisionService")
