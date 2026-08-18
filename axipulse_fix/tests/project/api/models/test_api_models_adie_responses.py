import importlib

def test_adie_responses_surface():
    module = importlib.import_module("api.models.adie_responses")
    assert hasattr(module, "ADIEDecisionResponse")
