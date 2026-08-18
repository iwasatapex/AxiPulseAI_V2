import importlib

def test_constraints_surface():
    module = importlib.import_module("core.forecast_ai.optimization.constraints")
    assert hasattr(module, "validate")
    assert hasattr(module, "validate_change")
    assert hasattr(module, "ConstraintValidator")
