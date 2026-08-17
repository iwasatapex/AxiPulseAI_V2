from typing import List
from .models import RiskFactor, RiskAnalysis
from .scoring import RiskScorer

class RiskAnalyzer:
    @staticmethod
    def analyze(component: str, risk_factors: List[RiskFactor]) -> RiskAnalysis:
        if not risk_factors:
            return RiskAnalysis(
                component=component,
                overall_risk=0.0,
                classification="Very Low",
                risk_factors=[],
                warnings=[],
                summary=f"No risks for {component}."
            )
        overall = RiskScorer.aggregate_risks(risk_factors)
        classification = RiskScorer.classify(overall)
        top = sorted(risk_factors, key=lambda r: r.risk_score, reverse=True)[:2]
        summary = f"{component}: {classification} ({overall:.2f}). " + ", ".join([f"{r.name} ({r.risk_score:.2f})" for r in top])
        return RiskAnalysis(
            component=component,
            overall_risk=overall,
            classification=classification,
            risk_factors=risk_factors,
            warnings=["High overall risk."] if overall > 0.5 else [],
            summary=summary
        )


# ---------------------------------------------------------------------------
# Module-level compatibility surface
# ---------------------------------------------------------------------------
def analyze(*args, **kwargs):
    return RiskAnalyzer.analyze(*args, **kwargs)
