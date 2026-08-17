import importlib

def test_exporter_surface():
    module = importlib.import_module("core.forecast_ai.reporting.exporter")
    assert hasattr(module, "to_dict")
    assert hasattr(module, "to_json")
    assert hasattr(module, "to_text")
    assert hasattr(module, "to_markdown")
    assert hasattr(module, "to_text_simple")
    assert hasattr(module, "ReportExporter")
