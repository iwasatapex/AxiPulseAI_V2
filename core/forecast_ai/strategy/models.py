from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class StrategyCategory(Enum):
    OPERATIONAL_EXCELLENCE = "operational_excellence"
    TRAINING = "training"
    CUSTOMER_EXPERIENCE = "customer_experience"
    QUALITY = "quality"
    STAFFING = "staffing"
    TECHNOLOGY = "technology"
    BALANCED = "balanced"
    RECOVERY = "recovery"
    PREVENTIVE = "preventive"
    GENERAL = "general"

@dataclass
class Milestone:
    week: int
    title: str
    description: str
    expected_progress: float = 0.0
    completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyPlan:
    id: str
    name: str
    description: str
    objective: str
    category: StrategyCategory
    priority: str  # Critical, High, Medium, Low
    estimated_operations_health: Optional[float] = None
    estimated_nps: Optional[float] = None
    estimated_duration_weeks: int = 4
    estimated_complexity: float = 0.5  # 0-1 scale
    estimated_disruption: float = 0.3   # 0-1 scale
    confidence: float = 0.7
    recommendations: List[str] = field(default_factory=list)  # recommendation IDs
    timeline: List[Milestone] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)  # alias for timeline
    risks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyResult:
    success: bool
    strategies: List[StrategyPlan]
    best_strategy: Optional[StrategyPlan] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
