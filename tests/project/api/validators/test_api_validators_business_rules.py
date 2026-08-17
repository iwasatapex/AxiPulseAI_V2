import importlib

def test_business_rules_surface():
    module = importlib.import_module("api.validators.business_rules")
    assert hasattr(module, "validate_business_rules")
