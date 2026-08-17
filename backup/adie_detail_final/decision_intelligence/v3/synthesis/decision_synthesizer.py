from dataclasses import dataclass
from typing import Any


@dataclass
class SynthesizedDecision:
    recommendation: str
    risk: str
    probability: float
    confidence: float
    expected: float
    downside: float
    upside: float


class DecisionSynthesizer:
    """ADIE 2.0 probabilistic decision synthesis layer."""

    def synthesize(
        self,
        scenarios: list[dict[str, Any]],
    ) -> SynthesizedDecision:
        if not scenarios:
            raise ValueError("scenarios must not be empty")

        ranked = sorted(
            scenarios,
            key=lambda x: (
                float(x.get("probability", 0.0)),
                float(x.get("confidence", 0.0)),
                float(x.get("expected", 0.0)),
            ),
            reverse=True,
        )

        best = ranked[0]

        probability = float(best.get("probability", 0.0))
        confidence = float(best.get("confidence", 0.0))
        expected = float(best.get("expected", 0.0))
        downside = float(best.get("p05", expected))
        upside = float(best.get("p95", expected))

        if probability >= 0.80 and confidence >= 0.70:
            risk = "LOW"
        elif probability >= 0.60 and confidence >= 0.55:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return SynthesizedDecision(
            recommendation=str(best.get("name", "current_state")),
            risk=risk,
            probability=probability,
            confidence=confidence,
            expected=expected,
            downside=downside,
            upside=upside,
        )


def synthesize(scenarios: list[dict[str, Any]]) -> SynthesizedDecision:
    """Module-level convenience: synthesize a decision from ranked scenarios."""
    return DecisionSynthesizer().synthesize(scenarios)


__all__ = ["DecisionSynthesizer", "SynthesizedDecision", "synthesize"]
