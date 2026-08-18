from core.decision_intelligence.v3.scenario.engine import (
    ADIEScenarioEngine,
    Scenario,
)


def test_scenario_execution():
    engine = ADIEScenarioEngine(samples=2000)

    scenario = Scenario(
        name="stable_operations",
        baseline=0.82,
        uncertainty=0.04,
        observations=[1, 1, 1, 0, 1],
    )

    result = engine.run(scenario)

    assert result.name == "stable_operations"
    assert 0.0 <= result.probability <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.p05 < result.p95
    assert result.samples == 2000


def test_scenario_comparison():
    engine = ADIEScenarioEngine(samples=2000)

    scenarios = [
        Scenario(
            name="baseline",
            baseline=0.75,
            uncertainty=0.05,
            observations=[1, 0, 1, 0, 1],
        ),
        Scenario(
            name="improved",
            baseline=0.90,
            uncertainty=0.03,
            observations=[1, 1, 1, 1, 1],
        ),
    ]

    results = engine.compare(scenarios)

    assert len(results) == 2
    assert results[0].name == "improved"


def test_scenario_serialization():
    engine = ADIEScenarioEngine(samples=1000)

    result = engine.run(
        Scenario(
            name="test",
            baseline=0.80,
            uncertainty=0.05,
            observations=[1, 1, 0, 1],
        )
    )

    payload = result.to_dict()

    assert payload["name"] == "test"
    assert "probability" in payload
    assert "p05" in payload
    assert "p95" in payload
