import importlib

def test_reverse_oh_solver_surface():
    module = importlib.import_module("reverse_oh_solver")
    assert hasattr(module, "load_latest_state")
    assert hasattr(module, "load_model")
    assert hasattr(module, "predict")
    assert hasattr(module, "solve")
