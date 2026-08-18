import importlib

def test_builder_surface():
    module = importlib.import_module("core.forecast_ai.reporting.builder")
    assert hasattr(module, "build")
    assert hasattr(module, "ReportBuilder")
