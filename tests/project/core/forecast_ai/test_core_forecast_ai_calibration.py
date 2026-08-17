import importlib

def test_calibration_surface():
    module = importlib.import_module("core.forecast_ai.calibration")
    assert hasattr(module, "summarize")
    assert hasattr(module, "ForecastCalibration")
