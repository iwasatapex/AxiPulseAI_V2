import importlib

def test_dashboard_formatter_surface():
    module = importlib.import_module("core.forecast_ai.reporting.dashboard_formatter")
    assert hasattr(module, "format_plain")
    assert hasattr(module, "format_ansi")
    assert hasattr(module, "DashboardFormatter")
