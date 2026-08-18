import importlib

def test_trace_surface():
    module = importlib.import_module("core.forecast_ai.explainability.trace")
    assert hasattr(module, "build_trace")
    assert hasattr(module, "TraceBuilder")
