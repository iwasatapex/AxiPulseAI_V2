from dataclasses import dataclass

from core.decision_intelligence.v3.policy import constants as C


@dataclass
class RiskAssessment:
    risk: str
    score: float
    confidence: float
    downside: float
    upside: float
    abstain: bool


class UncertaintyRiskEngine:
    """
    Canonical ADIE V3 risk model.

    This is the single risk semantic shared by every ADIE entry point
    (API and forecast). All thresholds/weights are centralized in
    ``policy.constants``; this class only applies them.
    """

    def assess(
        self,
        probability: float,
        confidence: float,
        downside: float,
        upside: float,
    ) -> RiskAssessment:
        probability = max(0.0, min(1.0, float(probability)))
        confidence = max(0.0, min(1.0, float(confidence)))
        downside = float(downside)
        upside = float(upside)

        weights = C.RISK_SCORE_WEIGHTS
        score = (
            (1.0 - probability) * weights["probability"]
            + (1.0 - confidence) * weights["confidence"]
            + max(0.0, probability - downside) * weights["downside"]
        )
        score = max(0.0, min(1.0, score))

        thresholds = C.RISK_THRESHOLDS
        if (
            confidence < thresholds["high_confidence"]
            or probability < thresholds["high_probability"]
        ):
            risk = "HIGH"
        elif (
            confidence < thresholds["medium_confidence"]
            or probability < thresholds["medium_probability"]
        ):
            risk = "MEDIUM"
        else:
            risk = "LOW"

        abstain_thresholds = C.ABSTAIN_THRESHOLDS
        abstain = confidence < abstain_thresholds["confidence"] or (
            probability < abstain_thresholds["probability"]
            and upside <= downside
        )

        return RiskAssessment(
            risk=risk,
            score=score,
            confidence=confidence,
            downside=downside,
            upside=upside,
            abstain=abstain,
        )


_default_risk_engine = UncertaintyRiskEngine()


def assess(
    probability: float,
    confidence: float,
    downside: float,
    upside: float,
    *,
    engine: "UncertaintyRiskEngine | None" = None,
) -> RiskAssessment:
    """Module-level convenience: run the canonical risk assessment."""
    return (engine or _default_risk_engine).assess(
        probability, confidence, downside, upside
    )


def risk_assessment_to_dict(risk: RiskAssessment) -> dict:
    """Serialize a RiskAssessment to a plain dict (for display/API)."""
    try:
        from dataclasses import asdict
        return asdict(risk)
    except Exception:
        return {
            "risk": getattr(risk, "risk", None),
            "score": getattr(risk, "score", None),
            "confidence": getattr(risk, "confidence", None),
            "downside": getattr(risk, "downside", None),
            "upside": getattr(risk, "upside", None),
            "abstain": getattr(risk, "abstain", None),
        }


__all__ = ["UncertaintyRiskEngine", "RiskAssessment", "assess", "risk_assessment_to_dict"]
