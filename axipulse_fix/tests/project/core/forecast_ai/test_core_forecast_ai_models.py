import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.models")
    assert hasattr(module, "OperationType")
    assert hasattr(module, "ScenarioType")
    assert hasattr(module, "ForecastRequest")
    assert hasattr(module, "ForecastResponse")
    assert hasattr(module, "ForecastDay")
    assert hasattr(module, "ForecastResult")
    assert hasattr(module, "ScenarioResult")
    assert hasattr(module, "Recommendation")
    assert hasattr(module, "Risk")
    assert hasattr(module, "Strategy")
    assert hasattr(module, "Confidence")
    assert hasattr(module, "Trend")
    assert hasattr(module, "PredictionRequest")
    assert hasattr(module, "PredictionResult")
