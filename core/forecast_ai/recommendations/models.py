from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class Category(Enum):
    TRAINING = "training"
    QUALITY = "quality"
    COMPETENCY = "competency"
    ATTENDANCE = "attendance"
    TRANSFER = "transfer"
    RELEASE = "release"
    OPERATIONS = "operations"
    CUSTOMER_EXPERIENCE = "customer_experience"
    STAFFING = "staffing"
    TECHNOLOGY = "technology"
    GENERAL = "general"

class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class Difficulty(Enum):
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"

@dataclass
class Recommendation:
    id: str
    title: str
    description: str
    category: Category
    priority: Priority
    difficulty: Difficulty
    estimated_operations_health_gain: Optional[float] = None
    estimated_nps_gain: Optional[float] = None
    estimated_disruption: float = 0.0  # 0-10 scale
    confidence: float = 0.5
    actions: List[str] = field(default_factory=list)
    reasoning: str = ""
    optimization_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Structured conflict-aware fields (Phase 10): prefer these over naive
    # keyword matching when reasoning about recommendation conflicts.
    target_kpi: Optional[str] = None       # e.g. "quality", "transfer"
    direction: Optional[str] = None        # "increase" | "decrease"
    magnitude: Optional[float] = None      # absolute proposed change

@dataclass
class RecommendationResult:
    success: bool
    recommendations: List[Recommendation]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Multiple generated-and-evaluated candidate scenarios (rank, predicted
    # vs target OH/NPS, feasibility, probabilistic interval, etc.) -- copied
    # through from OptimizationResult.metadata["ranked_candidates"] so GUI
    # consumers of RecommendationResult are not limited to a single
    # best-solution's per-field advice. Additive field; existing consumers
    # that only read `recommendations` are unaffected.
    candidates: List[Dict[str, Any]] = field(default_factory=list)
