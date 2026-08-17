import importlib

def test_test_report_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_report")
    assert hasattr(module, "setUp")
    assert hasattr(module, "test_executive_summary_synthesis")
    assert hasattr(module, "test_forecast_section_rich")
    assert hasattr(module, "test_appendix_generation")
    assert hasattr(module, "test_template_ordering")
    assert hasattr(module, "test_exporter_refactored")
    assert hasattr(module, "test_different_report_types")
    assert hasattr(module, "TestReportRefinements")
