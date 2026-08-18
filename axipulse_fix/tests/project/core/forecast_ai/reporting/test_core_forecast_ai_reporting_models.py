import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.reporting.models")
    assert hasattr(module, "ReportType")
    assert hasattr(module, "ReportSection")
    assert hasattr(module, "ExecutiveSummary")
    assert hasattr(module, "ReportAppendix")
    assert hasattr(module, "ReportMetadata")
    assert hasattr(module, "ReportResult")
