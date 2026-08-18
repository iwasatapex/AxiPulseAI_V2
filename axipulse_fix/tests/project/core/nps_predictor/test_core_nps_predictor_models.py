import importlib


def test_models_surface():
    module = importlib.import_module("core.nps_predictor.models")

    assert hasattr(module, "create_model_registry")
    assert hasattr(module, "compute_ensemble_weights")
    assert hasattr(module, "ModelRegistryMixin")
