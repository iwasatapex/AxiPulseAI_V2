from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class OperationType(Enum):
    FORECAST = "forecast"
    SCENARIO = "scenario"
    REVERSE_OPTIMIZE = "reverse_optimize"
    RECOMMEND = "recommend"
    STRATEGY = "strategy"
    TREND = "trend"
    SENSITIVITY = "sensitivity"
    CONFIDENCE = "confidence"
    RISK = "risk"
    EXPLAIN = "explain"
    REPORT = "report"

class ScenarioType(Enum):
    BASELINE = "baseline"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    AEP = "aep"
    OEP = "oep"
    TRAINING = "training"
    STAFFING_SHORTAGE = "staffing_shortage"
    TECHNOLOGY_UPGRADE = "technology_upgrade"
    CUSTOM = "custom"

@dataclass
class ForecastRequest:
    operation: Union[str, OperationType]
    target: Optional[str] = None
    horizon: Optional[int] = None
    scenario: Optional[Union[str, ScenarioType]] = None
    parameters: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ForecastResponse:
    success: bool
    operation: str
    engine: str
    timestamp: str
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]
    payload: Optional[Dict[str, Any]] = None

@dataclass
class ForecastDay:
    date: str
    operations_health: float
    nps: float
    quality: float
    competency: float
    transfer: float
    release: float
    attendance: float
    confidence: Optional[Dict[str, Any]] = None
    # Runtime: the Forecast-Risk engine emits a single risk dict.  (The field
    # is typed to also accept a list for legacy consumers that iterated it as
    # a list; the orchestrator stores the dict form.)
    risk: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    notes: Optional[str] = None
    # NPS 0..10 Bayesian posterior distribution produced by the NPS
    # predictor. Preserved so ADIE can consume it (expected NPS, P(NPS>=target),
    # score percentiles) without re-predicting. Additive; never fabricated.
    bayesian_score_distribution: Optional[Dict[str, float]] = None
    score_counts: Optional[Dict[str, int]] = None

@dataclass
class ForecastResult:
    horizon: int
    scenario: str
    start_date: str
    end_date: str
    timeline: List[ForecastDay]
    summary: Dict[str, Any]

@dataclass
class ScenarioResult:
    name: str
    description: str
    assumptions: Dict[str, Any]
    duration: int
    risk_level: str
    affected_kpis: List[str]
    forecast: Optional[ForecastResult] = None

@dataclass
class Recommendation:
    action: Dict[str, Any]
    expected_impact: float
    priority: int
    description: str

@dataclass
class Risk:
    kpi: str
    type: str
    severity: float
    message: str

@dataclass
class Strategy:
    name: str
    actions: Dict[str, Any]
    expected_outcome: Optional[Dict[str, float]] = None

@dataclass
class Confidence:
    expected: float
    lower_bound: float
    upper_bound: float
    score: float

@dataclass
class Trend:
    metric: str
    direction: str
    momentum: float
    volatility: float
    seasonality: Optional[Dict[str, Any]] = None

# ================================================================
# Prediction Layer Models
# ================================================================

@dataclass
class PredictionRequest:
    """Request model for PredictionService."""
    state: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PredictionResult:
    """Result model from PredictionService."""

    quality: Optional[float] = None
    competency: Optional[float] = None
    attendance: Optional[float] = None
    release: Optional[float] = None
    transfer: Optional[float] = None

    calls: Optional[int] = None

    operations_health: Optional[float] = None
    nps: Optional[float] = None

    confidence: Optional[float] = None

    # Real Bayesian 0–10 NPS posterior produced by core.nps_predictor.
    # These fields preserve the distribution through the production layer
    # instead of reducing the predictor result to a single NPS float.
    bayesian_score_distribution: Optional[Dict[str, float]] = None
    score_counts: Optional[Dict[str, int]] = None

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
