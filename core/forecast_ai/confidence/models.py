from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ConfidenceMetric:
    name: str
    score: float          # 0.0 – 1.0
    weight: float         # contribution weight
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfidenceAnalysis:
    component: str        # 'forecast', 'trend', 'sensitivity', 'recommendation', 'strategy'
    confidence_score: float
    classification: str   # Very High, High, Medium, Low, Very Low
    metrics: List[ConfidenceMetric]
    reasoning: str
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfidenceResult:
    success: bool
    overall_confidence: float
    forecast_confidence: Optional[ConfidenceAnalysis] = None
    trend_confidence: Optional[ConfidenceAnalysis] = None
    sensitivity_confidence: Optional[ConfidenceAnalysis] = None
    recommendation_confidence: Optional[ConfidenceAnalysis] = None
    strategy_confidence: Optional[ConfidenceAnalysis] = None
    analyses: List[ConfidenceAnalysis] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
