import importlib

def test_AxisPulseAI_surface():
    module = importlib.import_module("AxisPulseAI")
    assert hasattr(module, "banner")
    assert hasattr(module, "section")
    assert hasattr(module, "run")
    assert hasattr(module, "model_exists")
    assert hasattr(module, "do_train")
    assert hasattr(module, "do_predict")
    assert hasattr(module, "do_forecast")
    assert hasattr(module, "do_reverse")
    assert hasattr(module, "do_target_state")
    assert hasattr(module, "do_surprise")
    assert hasattr(module, "main_menu")
    assert hasattr(module, "ask")
    assert hasattr(module, "C")
