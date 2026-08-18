import pytest


from core.forecast_ai.prediction.production import ProductionPredictionAdapter


def _raw_nps_dict():
    return {
        "operations_health": 80.0,
        "nps": {
            "nps": 82.0,
            "total_surveys": 100,
            "score_counts": {f"score_{i}": (0 if i != 10 else 100) for i in range(11)},
            "bayesian_score_distribution": {f"score_{i}": (0.0 if i != 10 else 1.0) for i in range(11)},
        },
    }


class _Service:
    def __init__(self, raw):
        self.raw = raw

    def predict(self, request):
        return type("Raw", (), self.raw)()


def test_production_nps_uses_categorical_distribution():
    adapter = ProductionPredictionAdapter(service=_Service(_raw_nps_dict()))
    result = adapter.predict({}, simulations=250, seed=3)
    p = result.nps.probabilistic
    assert p.monte_carlo.metadata["distribution_domain"] == "survey_scores_0_10"
    assert p.monte_carlo.metadata["nps_derived_from_score_counts"] is True
    assert p.bayesian.metadata["scalar_nps_prediction_not_used_for_uncertainty"] is True
    assert p.likely_range_lower == 100.0
    assert p.likely_range_upper == 100.0


def test_production_nps_scalar_only_is_rejected():
    raw = {"operations_health": 80.0, "nps": 82.0}
    adapter = ProductionPredictionAdapter(service=_Service(raw))
    with pytest.raises(ValueError, match="scalar NPS uncertainty is prohibited"):
        adapter.predict({}, simulations=100)
