import importlib


def test_explainability_surface():
    module = importlib.import_module("core.nps_predictor.explainability")

    assert hasattr(module, "compute_shap")
    assert hasattr(module, "explain_nps")
