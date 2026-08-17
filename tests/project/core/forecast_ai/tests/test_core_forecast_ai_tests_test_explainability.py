import importlib

def test_test_explainability_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_explainability")
    assert hasattr(module, "test_inherited_confidence")
    assert hasattr(module, "test_forecast_uses_cross_component_data")
    assert hasattr(module, "test_trace_builder_includes_dependencies")
    assert hasattr(module, "test_evidence_reference")
    assert hasattr(module, "test_reasoning_builder_narrative")
    assert hasattr(module, "test_template_expansion")
    assert hasattr(module, "test_deterministic_explanation_id")
    assert hasattr(module, "TestExplainabilityImprovements")
