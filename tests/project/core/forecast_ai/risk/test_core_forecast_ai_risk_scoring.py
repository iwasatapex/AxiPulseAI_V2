import importlib

def test_scoring_surface():
    module = importlib.import_module("core.forecast_ai.risk.scoring")
    assert hasattr(module, "compute_risk_score")
    assert hasattr(module, "classify")
    assert hasattr(module, "aggregate_risks")
    assert hasattr(module, "RiskScorer")
