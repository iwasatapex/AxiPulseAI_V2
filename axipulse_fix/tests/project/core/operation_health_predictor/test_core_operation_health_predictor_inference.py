import importlib

def test_inference_surface():
    module = importlib.import_module("core.operation_health_predictor.inference")
    assert hasattr(module, "predict")
    assert hasattr(module, "predict_leaderboard")
    assert hasattr(module, "reverse_optimize")
    assert hasattr(module, "explain")
    assert hasattr(module, "make_input_vector")
    assert hasattr(module, "cost")
    assert hasattr(module, "realism_penalty")
    assert hasattr(module, "vector_to_dict")
    assert hasattr(module, "objective")
    assert hasattr(module, "InferenceMixin")
