from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

class ConstraintType(Enum):
    FIXED = "fixed"
    MAX = "max"
    MIN = "min"
    MAX_CHANGE = "max_change"

@dataclass
class Constraint:
    field: str
    type: ConstraintType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TargetGoal:
    target_operations_health: Optional[float] = None
    target_nps: Optional[float] = None
    deadline_days: Optional[int] = None
    tolerance: float = 0.5
    priority: str = "balanced"  # 'oh', 'nps', 'balanced'
    constraints: List[Constraint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationRequest:
    initial_state: Dict[str, float]
    target_goal: TargetGoal
    max_iterations: int = 100
    timeout_seconds: int = 30
    search_strategy: str = "hill_climb"
    seed: Optional[int] = None

@dataclass
class OptimizationSolution:
    predicted_operations_health: Optional[float]
    predicted_nps: Optional[float]
    state_changes: Dict[str, float]  # differences from original
    applied_scenarios: List[str] = field(default_factory=list)
    optimization_score: float = 0.0
    distance_to_target: float = 0.0
    iterations_used: int = 0
    constraints_satisfied: bool = True
    state: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    success: bool
    solutions: List[OptimizationSolution]
    best_solution: Optional[OptimizationSolution]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
