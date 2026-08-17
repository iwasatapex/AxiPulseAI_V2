from typing import Dict, Any, Optional
import math
from .models import TargetGoal, OptimizationSolution

class ScoreCalculator:
    @staticmethod
    def compute_distance(state: Dict[str, float], oh: Optional[float], nps: Optional[float],
                         target: TargetGoal) -> float:
        """Distance to an improvement target.

        Business targets are improvement goals ('reach at least X'). A metric
        that already meets or exceeds its target contributes zero distance;
        only a shortfall below the target moves distance away from zero.

        The previous symmetric (squared-Euclidean) distance penalized
        overshooting exactly as much as undershooting, so any state that was
        at-or-above the target looked far from it (e.g. OH 90.6 vs target 85
        -> distance 5.6 > tolerance 0.5). That made already-reached / easily
        reachable improvement targets report timeout or "no solution within
        tolerance". This directional form is the root-cause fix: overshoot is
        success, only shortfall is distance.
        """
        dist = 0.0
        if target.target_operations_health is not None and oh is not None:
            dist += max(0.0, float(target.target_operations_health) - float(oh)) ** 2
        if target.target_nps is not None and nps is not None:
            dist += max(0.0, float(target.target_nps) - float(nps)) ** 2
        return math.sqrt(dist)

    @staticmethod
    def compute_score(solution: OptimizationSolution, original_state: Dict[str, float],
                      target: TargetGoal) -> float:
        """Lower is better. Penalize large changes."""
        distance = solution.distance_to_target
        # Penalize magnitude of changes
        total_change = sum(abs(v) for v in solution.state_changes.values())
        # Normalize penalty
        penalty = total_change / 10.0  # arbitrary scaling
        return distance + penalty

    @staticmethod
    def is_acceptable(distance: float, target: TargetGoal) -> bool:
        return distance <= target.tolerance

# Module-level compatibility surface.
compute_distance = ScoreCalculator.compute_distance
compute_score = ScoreCalculator.compute_score
is_acceptable = ScoreCalculator.is_acceptable
