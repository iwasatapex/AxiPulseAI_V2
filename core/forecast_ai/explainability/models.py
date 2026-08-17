from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Evidence:
    component: str
    field: str
    value: Any
    importance: str          # High, Medium, Low
    description: str
    reference: Optional[str] = None   # e.g., 'forecast.timeline[5].operations_health'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExplanationTrace:
    step: int
    engine: str
    description: str
    purpose: str
    input_reference: str
    output_reference: str
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Explanation:
    id: str
    title: str
    component: str
    summary: str
    reasoning: str
    evidence: List[Evidence]
    conclusion: str
    confidence: Optional[float] = None   # inherited, not hardcoded
    source_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExplainabilityResult:
    success: bool
    overall_summary: str
    forecast_explanation: Optional[Explanation] = None
    trend_explanation: Optional[Explanation] = None
    sensitivity_explanation: Optional[Explanation] = None
    recommendation_explanation: Optional[Explanation] = None
    strategy_explanation: Optional[Explanation] = None
    confidence_explanation: Optional[Explanation] = None
    risk_explanation: Optional[Explanation] = None
    traces: List[ExplanationTrace] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
