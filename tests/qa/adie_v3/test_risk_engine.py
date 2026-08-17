from core.decision_intelligence.v3.risk.uncertainty import (
    UncertaintyRiskEngine,
)


def test_low_risk():
    engine = UncertaintyRiskEngine()

    result = engine.assess(
        probability=0.8571,
        confidence=0.7526,
        downside=0.8494,
        upside=0.9485,
    )

    assert result.risk == "LOW"
    assert result.abstain is False
    assert 0.0 <= result.score <= 1.0


def test_high_risk():
    engine = UncertaintyRiskEngine()

    result = engine.assess(
        probability=0.25,
        confidence=0.20,
        downside=0.10,
        upside=0.40,
    )

    assert result.risk == "HIGH"
    assert result.abstain is True


def test_bounds():
    engine = UncertaintyRiskEngine()

    result = engine.assess(
        probability=2.0,
        confidence=-1.0,
        downside=0.2,
        upside=0.8,
    )

    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
