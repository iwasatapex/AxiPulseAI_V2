import importlib

def test_report_engine_surface():
    module = importlib.import_module("core.forecast_ai.engines.report_engine")
    assert hasattr(module, "execute")
    assert hasattr(module, "ReportEngine")
