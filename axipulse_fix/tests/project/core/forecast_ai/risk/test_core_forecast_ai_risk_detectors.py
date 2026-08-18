import importlib

def test_detectors_surface():
    module = importlib.import_module("core.forecast_ai.risk.detectors")
    assert hasattr(module, "detect")
    assert hasattr(module, "detect")
    assert hasattr(module, "detect")
    assert hasattr(module, "detect")
    assert hasattr(module, "detect")
    assert hasattr(module, "detect")
    assert hasattr(module, "ForecastRiskDetector")
    assert hasattr(module, "TrendRiskDetector")
    assert hasattr(module, "SensitivityRiskDetector")
    assert hasattr(module, "RecommendationRiskDetector")
    assert hasattr(module, "StrategyRiskDetector")
    assert hasattr(module, "ConfidenceRiskDetector")
