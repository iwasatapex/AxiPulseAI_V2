"""
ExperimentGenerator – creates controlled modifications to each KPI.
Generates both +Δ and −Δ experiments for symmetrical sensitivity.
"""
from typing import List, Dict, Any, Optional
from copy import deepcopy
from ..config import KPI_BOUNDS

class ExperimentGenerator:
    @staticmethod
    def generate_experiments(state: Dict[str, float],
                             step_size: float = 1.0,
                             metrics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate experiments for each KPI: one with +step_size, one with -step_size.
        Uses KPI_BOUNDS from config to clamp values.
        Returns list of experiments with keys: 'metric', 'baseline_state', 'modified_state', 'delta'.
        """
        if metrics is None:
            metrics = ['quality', 'competency', 'attendance', 'release', 'transfer']

        experiments = []
        baseline = deepcopy(state)

        for metric in metrics:
            if metric not in state:
                continue
            # Get bounds from config
            bounds = KPI_BOUNDS.get(metric, (0, 100))
            min_val, max_val = bounds

            # Positive perturbation
            modified_plus = deepcopy(baseline)
            new_val = modified_plus[metric] + step_size
            modified_plus[metric] = max(min_val, min(max_val, new_val))
            delta_plus = modified_plus[metric] - baseline[metric]

            # Negative perturbation
            modified_minus = deepcopy(baseline)
            new_val = modified_minus[metric] - step_size
            modified_minus[metric] = max(min_val, min(max_val, new_val))
            delta_minus = modified_minus[metric] - baseline[metric]

            # Only add if delta is non-zero (i.e., not at boundary)
            if abs(delta_plus) > 1e-6:
                experiments.append({
                    'metric': metric,
                    'direction': '+',
                    'baseline_state': baseline,
                    'modified_state': modified_plus,
                    'delta': delta_plus
                })
            if abs(delta_minus) > 1e-6:
                experiments.append({
                    'metric': metric,
                    'direction': '-',
                    'baseline_state': baseline,
                    'modified_state': modified_minus,
                    'delta': delta_minus
                })
        return experiments
