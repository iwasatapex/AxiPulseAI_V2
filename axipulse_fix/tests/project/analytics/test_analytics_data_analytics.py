import importlib

def test_data_analytics_surface():
    module = importlib.import_module("analytics.data_analytics")
    assert hasattr(module, "quality_report")
    assert hasattr(module, "outlier_detection")
    assert hasattr(module, "summary_stats")
    assert hasattr(module, "correlation_matrix")
    assert hasattr(module, "DataAnalytics")
