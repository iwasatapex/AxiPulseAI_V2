import importlib

def test_commands_surface():
    module = importlib.import_module("cli.commands")
    assert hasattr(module, "cprint")
    assert hasattr(module, "render_full_dashboard")
    assert hasattr(module, "run_train")
    assert hasattr(module, "run_predict")
    assert hasattr(module, "run_predict_nps")
    assert hasattr(module, "run_reverse_nps")
    assert hasattr(module, "run_explain_nps")
    assert hasattr(module, "run_leaderboard_nps")
    assert hasattr(module, "run_defaults")
    assert hasattr(module, "main")
    assert hasattr(module, "render_full_dashboard")
    assert hasattr(module, "run_train_nps_predictor")
    assert hasattr(module, "cprint")
    assert hasattr(module, "left_print")
