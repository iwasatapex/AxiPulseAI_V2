import importlib

def test_decision_synthesizer_surface():
    module = importlib.import_module("core.decision_intelligence.v3.synthesis.decision_synthesizer")
    assert hasattr(module, "synthesize")
    assert hasattr(module, "SynthesizedDecision")
    assert hasattr(module, "DecisionSynthesizer")
