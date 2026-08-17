import importlib

def test_trainer_surface():
    module = importlib.import_module("core.operation_health_predictor.trainer")
    assert hasattr(module, "train")
    assert hasattr(module, "get_feature_importance")
    assert hasattr(module, "evaluate_fold")
    assert hasattr(module, "TrainerMixin")
