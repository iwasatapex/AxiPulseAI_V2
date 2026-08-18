from core.forecast_ai.prediction.service import PredictionService


class StubOH:
    def predict(self, state):
        return 91.5


class StubNPS:
    def predict(self, state):
        return {"nps": 84.0, "bayesian_score_distribution": {}, "score_counts": {}}


def test_injected_only_oh_does_not_call_missing_nps():
    service = PredictionService(oh_predictor=StubOH())

    request = type(
        "Request",
        (),
        {
            "state": {
                "quality": 87,
                "competency": 93,
                "attendance": 90,
                "release": 60,
                "transfer": 9,
            },
            "metadata": {},
        },
    )()

    result = service.predict(request)

    assert result.operations_health == 91.5
    assert result.nps is None
    assert not any("NPS prediction error" in e for e in result.errors)


def test_injected_only_nps_does_not_call_missing_oh():
    service = PredictionService(nps_predictor=StubNPS())

    request = type(
        "Request",
        (),
        {
            "state": {
                "quality": 87,
                "competency": 93,
                "attendance": 90,
                "release": 60,
                "transfer": 9,
            },
            "metadata": {},
        },
    )()

    result = service.predict(request)

    assert result.operations_health is None
    assert result.nps == 84.0
    assert not any("OH prediction error" in e for e in result.errors)
