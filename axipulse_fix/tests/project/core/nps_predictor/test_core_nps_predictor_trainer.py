import importlib

def test_trainer_surface():
    module = importlib.import_module("core.nps_predictor.trainer")
    assert hasattr(module, "tqdm_joblib")
    assert hasattr(module, "train_nps_predictor")
    assert hasattr(module, "load_data")
    assert hasattr(module, "cold_start_train")
    assert hasattr(module, "rolling_origin_train")
    assert hasattr(module, "train_and_select_simple")
