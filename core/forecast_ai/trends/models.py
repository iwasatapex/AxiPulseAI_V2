from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class TrendSeries:
    """A time series for a single KPI."""
    metric: str
    values: List[float]
    timestamps: List[str]  # ISO date strings
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendAnalysis:
    """Analysis result for a single KPI."""
    metric: str
    trend_direction: str  # Strong Increase, Increase, Stable, Decrease, Strong Decrease
    trend_strength: str   # Weak, Moderate, Strong
    moving_average: List[float]
    minimum: float
    maximum: float
    mean: float
    median: float
    variance: float
    standard_deviation: float
    volatility: str  # Low, Medium, High
    absolute_change: float
    percent_change: float
    pattern: str  # Increasing, Decreasing, Plateau, Recovery, Oscillation, Spike, Stable
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendResult:
    success: bool
    analyses: List[TrendAnalysis]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
