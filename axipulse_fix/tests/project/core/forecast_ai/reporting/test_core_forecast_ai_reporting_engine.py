import importlib

def test_engine_surface():
    module = importlib.import_module("core.forecast_ai.reporting.engine")
    assert hasattr(module, "generate")
    assert hasattr(module, "export_json")
    assert hasattr(module, "export_markdown")
    assert hasattr(module, "export_text")
    assert hasattr(module, "export_dict")
    assert hasattr(module, "ReportEngine")
