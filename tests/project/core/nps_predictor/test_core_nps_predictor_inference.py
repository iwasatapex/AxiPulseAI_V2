import importlib


def test_inference_surface():
    module = importlib.import_module("core.nps_predictor.inference")

    assert hasattr(module, "fallback_predict")
    assert hasattr(module, "postprocess_predictions")
    assert hasattr(module, "predict_single")
    assert hasattr(module, "predict_ensemble")
    assert hasattr(module, "predict_leaderboard")
