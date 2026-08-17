import importlib

def test_business_analytics_surface():
    module = importlib.import_module("analytics.business_analytics")
    assert hasattr(module, "nps_summary")
    assert hasattr(module, "nps_trend")
    assert hasattr(module, "oh_summary")
    assert hasattr(module, "kpi_gap_analysis")
    assert hasattr(module, "survey_distribution")
    assert hasattr(module, "call_patterns")
    assert hasattr(module, "release_transfer_analysis")
    assert hasattr(module, "BusinessAnalytics")
