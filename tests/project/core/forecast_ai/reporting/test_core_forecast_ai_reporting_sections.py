import importlib

def test_sections_surface():
    module = importlib.import_module("core.forecast_ai.reporting.sections")
    assert hasattr(module, "forecast_section")
    assert hasattr(module, "trend_section")
    assert hasattr(module, "sensitivity_section")
    assert hasattr(module, "recommendations_section")
    assert hasattr(module, "strategy_section")
    assert hasattr(module, "confidence_section")
    assert hasattr(module, "risk_section")
    assert hasattr(module, "explainability_section")
    assert hasattr(module, "appendix_section")
    assert hasattr(module, "SectionGenerator")
