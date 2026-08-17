"""API/forecast decision parity (Phase 6).

For identical decision inputs, the API scenario path and the forecast path
(ProbabilisticDecisionService) must produce semantically equivalent risk and
decision semantics — the single canonical decision model.
"""

from api.services.adie_v3_service import ADIEV3Service
from core.decision_intelligence.v3.integration.probabilistic_decision import (
    ProbabilisticDecisionService,
)


SCENARIOS = [
    {"name": "current_state", "probability": 0.6, "confidence": 0.6,
     "expected": 0.80, "p05": 0.70, "p95": 0.90},
    {"name": "improved_operations", "probability": 0.85, "confidence": 0.75,
     "expected": 0.90, "p05": 0.84, "p95": 0.95},
]


def test_api_scenario_and_forecast_service_share_risk_semantics():
    service = ADIEV3Service()
    api_pkg = service.analyze_scenarios(
        scenarios=SCENARIOS,
        observations=[1, 1, 0, 1, 1],
        baseline=0.82,
        samples=2000,
    )
    forecast_pkg = ProbabilisticDecisionService().analyze(
        scenarios=[dict(s) for s in SCENARIOS],
        observations=[1, 1, 0, 1, 1],
        baseline=0.82,
        samples=2000,
    )
    # Same canonical risk model and same best-scenario semantics.
    assert api_pkg.risk == forecast_pkg.risk
    assert api_pkg.recommendation == forecast_pkg.recommendation
    assert api_pkg.risk_score == forecast_pkg.risk_score
    assert api_pkg.abstain == forecast_pkg.abstain
    assert api_pkg.decision["action"] == forecast_pkg.decision["action"]


def test_api_plain_path_uses_canonical_risk_and_meaningful_decision():
    service = ADIEV3Service()
    low = service.analyze(observations=[1, 1, 1, 1, 1], baseline=0.9, samples=1000)
    high = service.analyze(observations=[0, 0, 0, 0, 0], baseline=0.1, samples=1000)
    assert low["risk"]["risk"] == "LOW"
    assert high["risk"]["risk"] == "HIGH"
    # Decisions vary by input (no hard-coded recommendation).
    assert low["decision"]["recommendation"] != high["decision"]["recommendation"]
    assert low["decision"]["action"] != high["decision"]["action"]
