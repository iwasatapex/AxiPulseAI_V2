import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.optimization.models")
    assert hasattr(module, "ConstraintType")
    assert hasattr(module, "Constraint")
    assert hasattr(module, "TargetGoal")
    assert hasattr(module, "OptimizationRequest")
    assert hasattr(module, "OptimizationSolution")
    assert hasattr(module, "OptimizationResult")
