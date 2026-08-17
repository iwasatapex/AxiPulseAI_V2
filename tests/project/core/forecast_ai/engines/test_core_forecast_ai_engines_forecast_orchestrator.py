import importlib

def test_forecast_orchestrator_surface():
    module = importlib.import_module("core.forecast_ai.engines.forecast_orchestrator")
    assert hasattr(module, "execute")
    assert hasattr(module, "regenerate_forecast")
    assert hasattr(module, "update_actual")
    assert hasattr(module, "ForecastOrchestrator")
