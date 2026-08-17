from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SensitivityExperiment:
    """A single experiment: change one KPI by delta, measure output."""
    metric: str                 # e.g., 'quality'
    baseline_value: float
    modified_value: float
    delta: float                # modified - baseline
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SensitivityAnalysis:
    """Results of a sensitivity experiment."""
    metric: str
    baseline_output_oh: float
    baseline_output_nps: float
    modified_output_oh: float
    modified_output_nps: float
    operations_health_change: float   # absolute change
    nps_change: float                 # absolute change
    sensitivity_score_oh: float       # dOH / dInput
    sensitivity_score_nps: float      # dNPS / dInput
    elasticity_oh: float              # %dOH / %dInput
    elasticity_nps: float             # %dNPS / %dInput
    rank: int = 0
    classification: str = "Medium"    # Very High, High, Medium, Low, Negligible
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SensitivityResult:
    success: bool
    analyses: List[SensitivityAnalysis]
    ranking: List[SensitivityAnalysis] = field(default_factory=list)  # sorted by sensitivity (highest first)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
