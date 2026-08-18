import importlib


def test_inference_surface():
    module = importlib.import_module("core.nps_predictor.inference")

    assert hasattr(module, "fallback_predict")
    assert hasattr(module, "postprocess_predictions")
    assert hasattr(module, "predict_single")
    assert hasattr(module, "predict_ensemble")
    assert hasattr(module, "predict_leaderboard")


def test_fallback_predict_never_fabricates_scalar_nps_confidence():
    """Degraded-mode fallback must not emit a scalar NPS ± confidence band.

    In the absence of a model artifact there is no 0..10 score distribution, so
    the NPS uncertainty invariant forbids fabricating a ± band from the scalar
    NPS. The interval must be a point-only zero-width band, and no fabricated
    scalar ``confidence`` may be emitted.
    """
    module = importlib.import_module("core.nps_predictor.inference")

    row = {
        "total_calls_received": 100,
        "actual_release_rate": 60.0,
        "operational_health": 80.0,
    }
    result = module.fallback_predict(None, row)

    nps = result["nps"]
    # Point-only interval: no scalar-NPS ± confidence width.
    assert result["prediction_interval"] == {"low": float(nps), "high": float(nps)}
    # No fabricated scalar confidence value.
    assert "confidence" not in result
