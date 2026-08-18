from core.decision_intelligence.v3.synthesis.decision_synthesizer import (
    DecisionSynthesizer,
)


def test_synthesis_selects_best_scenario():
    engine = DecisionSynthesizer()

    result = engine.synthesize([
        {
            "name": "current_state",
            "probability": 0.7143,
            "confidence": 0.6806,
            "expected": 0.8192,
            "p05": 0.7525,
            "p95": 0.8846,
        },
        {
            "name": "improved_operations",
            "probability": 0.8571,
            "confidence": 0.7526,
            "expected": 0.8994,
            "p05": 0.8494,
            "p95": 0.9485,
        },
    ])

    assert result.recommendation == "improved_operations"
    assert result.risk == "LOW"
    assert 0.0 <= result.probability <= 1.0
    assert result.downside < result.upside


def test_empty_scenarios_rejected():
    engine = DecisionSynthesizer()

    try:
        engine.synthesize([])
        assert False
    except ValueError:
        assert True
