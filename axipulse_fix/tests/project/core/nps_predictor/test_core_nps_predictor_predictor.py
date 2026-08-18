import importlib


def test_predictor_surface():
    module = importlib.import_module("core.nps_predictor.predictor")

    assert hasattr(module, "NPSPredictor")
    assert hasattr(module, "predict_ensemble")
    assert hasattr(module, "predict_leaderboard")
    assert hasattr(module, "explain_nps")
    assert hasattr(module, "reverse_optimize_nps")
    assert hasattr(module, "detect_drift")
    assert hasattr(module, "save_model")
    assert hasattr(module, "load_model")
    assert hasattr(module, "train_nps_predictor")
