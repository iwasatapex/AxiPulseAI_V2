import logging
from typing import Dict, Any, Optional
from .models import RiskResult
from .detectors import (
    ForecastRiskDetector, TrendRiskDetector, SensitivityRiskDetector,
    RecommendationRiskDetector, StrategyRiskDetector, ConfidenceRiskDetector
)
from .analyzer import RiskAnalyzer
from ..config import RISK_WEIGHTS

logger = logging.getLogger(__name__)

class RiskEngine:
    def __init__(self):
        self.analyzer = RiskAnalyzer()

    def evaluate(self,
                 forecast_result: Optional[Dict] = None,
                 trend_result: Optional[Any] = None,
                 sensitivity_result: Optional[Any] = None,
                 recommendation_result: Optional[Any] = None,
                 strategy_result: Optional[Any] = None,
                 confidence_result: Optional[Any] = None) -> RiskResult:
        components = {
            'forecast': (forecast_result, ForecastRiskDetector.detect),
            'trend': (trend_result, TrendRiskDetector.detect),
            'sensitivity': (sensitivity_result, SensitivityRiskDetector.detect),
            'recommendation': (recommendation_result, RecommendationRiskDetector.detect),
            'strategy': (strategy_result, StrategyRiskDetector.detect),
            'confidence': (confidence_result, ConfidenceRiskDetector.detect),
        }
        analyses = {}
        warnings = []
        errors = []
        for name, (result, detector) in components.items():
            if result is not None:
                try:
                    risks = detector(result)
                    if risks:
                        analyses[name] = self.analyzer.analyze(name, risks)
                        warnings.extend(analyses[name].warnings)
                except Exception as e:
                    errors.append(f"{name} error: {str(e)}")
        if not analyses:
            return RiskResult(success=False, overall_risk=0.0, warnings=warnings, errors=errors or ["No risk analyses."])
        total = sum(RISK_WEIGHTS.get(n, 0) for n in analyses)
        weighted = sum(analyses[n].overall_risk * RISK_WEIGHTS.get(n, 0) for n in analyses)
        overall = weighted / total if total > 0 else 0.0
        return RiskResult(
            success=True,
            overall_risk=overall,
            forecast_risk=analyses.get('forecast'),
            trend_risk=analyses.get('trend'),
            sensitivity_risk=analyses.get('sensitivity'),
            recommendation_risk=analyses.get('recommendation'),
            strategy_risk=analyses.get('strategy'),
            confidence_risk=analyses.get('confidence'),
            analyses=list(analyses.values()),
            warnings=warnings,
            errors=errors,
            metadata={"num_components": len(analyses)}
        )


# ---------------------------------------------------------------------------
# Module-level compatibility surface
# ---------------------------------------------------------------------------
def evaluate(*args, **kwargs):
    return RiskEngine().evaluate(*args, **kwargs)
