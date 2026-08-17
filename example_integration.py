from core.target_state_engine.engine import TargetStateEngine
from core.decision_intelligence.v3.integration.probabilistic_decision import (
    ProbabilisticDecisionService,
)


def main():
    # Simulated upstream outputs
    forecast_output = {
        "predicted_oh": 78.5,
        "transfer_trend": 1.4,
        "complexity_index": 0.72,
    }
    anic_output = {
        "predictions": {"oh": 78.9, "nps": 42},
        "confidence": 88,
        "consensus": True,
        "agreement_score": 0.91,
    }

    # Use the real TSE API
    tse = TargetStateEngine()
    tse_output = tse.find_target_state(
        {
            "operational_health": 88.0,
            "nps": 50.0
        }
    )

    # Canonical ADIE V3 decision pipeline (advisor-only).
    service = ProbabilisticDecisionService()
    baseline = float(forecast_output.get("predicted_oh", 80.0)) / 100.0
    package = service.analyze(
        scenarios=[
            {
                "name": "current_state",
                "expected": baseline,
            }
        ],
        observations=[
            baseline,
            float(anic_output.get("agreement_score", 0.9)),
        ],
        baseline=baseline,
        samples=5000,
    )

    print(f"Recommendation: {package.recommendation}")
    print(f"Risk: {package.risk} (probability {package.probability:.3f}, confidence {package.confidence:.3f})")
    print(f"Expected: {package.expected:.3f}  downside={package.downside:.3f}  upside={package.upside:.3f}")


if __name__ == "__main__":
    main()
