import importlib

def test_decision_policy_surface():
    module = importlib.import_module("core.decision_intelligence.v3.policy.decision_policy")
    assert hasattr(module, "select")
    assert hasattr(module, "PolicyDecision")
    assert hasattr(module, "DecisionPolicyEngine")
