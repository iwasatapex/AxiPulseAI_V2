import importlib

def test_input_surface():
    module = importlib.import_module("cli.input")
    assert hasattr(module, "get_input")
    assert hasattr(module, "collect_day_inputs")
