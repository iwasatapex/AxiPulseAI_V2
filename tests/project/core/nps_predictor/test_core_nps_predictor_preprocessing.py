import importlib

def test_preprocessing_surface():
    module = importlib.import_module("core.nps_predictor.preprocessing")
    assert hasattr(module, "compute_feature_stats")
    assert hasattr(module, "impute_missing")
    assert hasattr(module, "clip_outliers_iqr")
