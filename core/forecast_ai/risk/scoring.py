from typing import List
from .models import RiskFactor
from ..config import RISK_THRESHOLDS, RISK_SCORE_WEIGHTS, COMPONENT_RISK_AGGREGATION

class RiskScorer:
    @staticmethod
    def compute_risk_score(severity: float, probability: float, impact: float) -> float:
        return (severity * RISK_SCORE_WEIGHTS['severity'] +
                probability * RISK_SCORE_WEIGHTS['probability'] +
                impact * RISK_SCORE_WEIGHTS['impact'])

    @staticmethod
    def classify(score: float) -> str:
        if score >= RISK_THRESHOLDS['critical']:
            return "Critical"
        elif score >= RISK_THRESHOLDS['high']:
            return "High"
        elif score >= RISK_THRESHOLDS['medium']:
            return "Medium"
        elif score >= RISK_THRESHOLDS['low']:
            return "Low"
        else:
            return "Very Low"

    @staticmethod
    def aggregate_risks(risk_factors: List[RiskFactor]) -> float:
        if not risk_factors:
            return 0.0
        scores = [r.risk_score for r in risk_factors]
        if COMPONENT_RISK_AGGREGATION == 'max':
            return max(scores)
        elif COMPONENT_RISK_AGGREGATION == 'weighted_average':
            return sum(scores) / len(scores)
        elif COMPONENT_RISK_AGGREGATION == 'top3_average':
            top3 = sorted(scores, reverse=True)[:3]
            return sum(top3) / len(top3)
        else:
            return max(scores)


# ---------------------------------------------------------------------------
# Module-level compatibility surface
# ---------------------------------------------------------------------------
def compute_risk_score(*args, **kwargs):
    return RiskScorer.compute_risk_score(*args, **kwargs)

def classify(*args, **kwargs):
    return RiskScorer.classify(*args, **kwargs)

def aggregate_risks(*args, **kwargs):
    return RiskScorer.aggregate_risks(*args, **kwargs)
