from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class RiskCategory(Enum):
    FORECAST = "forecast"
    OPERATIONAL = "operational"
    STRATEGY = "strategy"
    RECOMMENDATION = "recommendation"
    TREND = "trend"
    SENSITIVITY = "sensitivity"
    CONFIDENCE = "confidence"
    STAFFING = "staffing"
    TECHNOLOGY = "technology"
    CUSTOMER_EXPERIENCE = "customer_experience"
    QUALITY = "quality"
    ATTENDANCE = "attendance"
    TRANSFER = "transfer"
    RELEASE = "release"
    GENERAL = "general"

@dataclass
class RiskFactor:
    id: str
    name: str
    category: RiskCategory
    severity: float
    probability: float
    impact: float
    risk_score: float
    reason: str
    mitigation: str
    source: str
    # Risk SOURCE contract: how severity/probability/impact were produced.
    #   "business_rule" -> deterministic hardcoded thresholds/rules
    #   "model"         -> derived from an actual model output
    #   "derived"       -> a metric computed from data (e.g. CV volatility)
    # Downstream (GUI / decision) must never label a business_rule severity
    # as an ML prediction.
    source_kind: str = "business_rule"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Centralized source attribution validation: a model-derived risk must
        # declare source_kind="model" (it must NOT be a detector-name source
        # silently defaulted to business_rule), and a business-rule risk must
        # not masquerade as model-derived.
        validate_risk_source_kind(self)


def validate_risk_source_kind(risk: "RiskFactor") -> None:
    """Validate a RiskFactor's source attribution is self-consistent.

    - ``source_kind == "model"`` requires the ``source`` to reference an actual
      model output (e.g. a model name / metric), and the factor must carry a
      model-output provenance in ``metadata`` (``model_output``).
    - ``source_kind in {"business_rule", "derived"}`` must NOT claim to be an
      ML prediction (no ``model_output`` provenance).

    Raises ``ValueError`` on contradictory attribution.
    """
    kind = (risk.source_kind or "").strip().lower()
    if kind not in {"business_rule", "model", "derived"}:
        raise ValueError(
            f"Risk factor {risk.id!r} has invalid source_kind={risk.source_kind!r}. "
            "Must be one of business_rule/model/derived."
        )
    has_model_output = bool(risk.metadata.get("model_output"))
    if kind == "model" and not has_model_output:
        raise ValueError(
            f"Risk factor {risk.id!r} declares source_kind='model' but carries "
            "no model_output provenance in metadata; a model-derived severity "
            "must trace to an actual model output."
        )
    if kind != "model" and has_model_output:
        raise ValueError(
            f"Risk factor {risk.id!r} has source_kind={kind!r} but carries "
            "model_output provenance; it must declare source_kind='model'."
        )

@dataclass
class RiskAnalysis:
    component: str
    overall_risk: float
    classification: str
    risk_factors: List[RiskFactor]
    warnings: List[str] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskResult:
    success: bool
    overall_risk: float
    forecast_risk: Optional[RiskAnalysis] = None
    trend_risk: Optional[RiskAnalysis] = None
    sensitivity_risk: Optional[RiskAnalysis] = None
    recommendation_risk: Optional[RiskAnalysis] = None
    strategy_risk: Optional[RiskAnalysis] = None
    confidence_risk: Optional[RiskAnalysis] = None
    analyses: List[RiskAnalysis] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
