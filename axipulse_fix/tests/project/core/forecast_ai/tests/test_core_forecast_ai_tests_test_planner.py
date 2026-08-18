import importlib

def test_test_planner_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_planner")
    assert hasattr(module, "test_forecast_request_success")
    assert hasattr(module, "test_forecast_without_state_fails")
    assert hasattr(module, "test_invalid_operation")
    assert hasattr(module, "test_missing_operation")
    assert hasattr(module, "setUp")
    assert hasattr(module, "test_router_returns_correct_engine")
    assert hasattr(module, "TestPlannerEndToEnd")
    assert hasattr(module, "TestRouting")
