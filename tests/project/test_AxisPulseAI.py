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
    assert hasattr(module, "do_surprise")
    assert hasattr(module, "main_menu")
    assert hasattr(module, "ask")
    assert hasattr(module, "C")
    # The legacy Target State Engine user-facing path has been removed; the
    # CLI now routes reverse/target-state exclusively through ReverseOptimizer.
    assert not hasattr(module, "do_target_state")
