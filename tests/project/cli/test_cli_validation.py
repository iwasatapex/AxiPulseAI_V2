import importlib

def test_validation_surface():
    module = importlib.import_module("cli.validation")
    assert hasattr(module, "validate_business_rules")
