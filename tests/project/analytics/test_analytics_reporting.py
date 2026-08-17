import importlib

def test_reporting_surface():
    module = importlib.import_module("analytics.reporting")
    assert hasattr(module, "generate_full_report")
    assert hasattr(module, "add_results")
    assert hasattr(module, "to_json")
    assert hasattr(module, "to_csv")
    assert hasattr(module, "to_html")
    assert hasattr(module, "print_summary")
    assert hasattr(module, "ReportGenerator")
