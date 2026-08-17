"""
SensitivityEngine – orchestrates experiments and analysis.
Runs +Δ and −Δ experiments, then aggregates.
"""
import logging
from typing import Dict, Any, List, Optional
from .models import SensitivityResult
from .experiments import ExperimentGenerator
from .analyzer import SensitivityAnalyzer
from .ranking import SensitivityRanker
from ..prediction import PredictionService
from ..models import PredictionRequest
from ..state import OperationalState
from ..config import KPI_BOUNDS

logger = logging.getLogger(__name__)

class SensitivityEngine:
    def __init__(self, prediction_service: Optional[PredictionService] = None,
                 step_size: float = 1.0):
        self.service = prediction_service or PredictionService()
        self.step_size = step_size
        self.generator = ExperimentGenerator()
        self.analyzer = SensitivityAnalyzer()
        self.ranker = SensitivityRanker()

    def analyze(self, state: Dict[str, float],
                metrics: Optional[List[str]] = None) -> SensitivityResult:
        if not state:
            return SensitivityResult(
                success=False,
                analyses=[],
                ranking=[],
                errors=["Empty state provided."]
            )

        # Generate experiments (both + and -)
        experiments = self.generator.generate_experiments(state, self.step_size, metrics)

        if not experiments:
            return SensitivityResult(
                success=False,
                analyses=[],
                ranking=[],
                errors=["No experiments could be generated."]
            )

        # Get baseline predictions (batched with all experiments below).
        # Build the list of states: baseline first, then each experiment.
        experiment_states = [
            (exp['metric'], exp.get('direction', '+'), exp['delta'], exp['modified_state'])
            for exp in experiments
        ]

        all_states = [dict(state)] + [ms for _, _, _, ms in experiment_states]

        try:
            batch_results = self.service.predict_batch(all_states)
        except Exception as e:
            # Fall back to the original per-prediction path if batching fails.
            batch_results = None

        def _get_pred(idx):
            if batch_results is not None and idx < len(batch_results):
                r = batch_results[idx]
                oh = r.get('operations_health')
                nps = r.get('nps')
                if oh is None:
                    oh = 0.0
                if nps is None:
                    nps = 0.0
                return {'operations_health': oh, 'nps': nps, 'errors': r.get('errors', [])}
            return None

        baseline = _get_pred(0)
        if baseline is None:
            # Fall back to per-state prediction for baseline.
            try:
                baseline_pred = self._predict(state)
                baseline_oh = baseline_pred.get('operations_health')
                baseline_nps = baseline_pred.get('nps')
                if baseline_oh is None or baseline_nps is None:
                    baseline_oh = 0.0
                    baseline_nps = 0.0
            except Exception as e:
                return SensitivityResult(
                    success=False,
                    analyses=[],
                    ranking=[],
                    errors=[f"Baseline prediction error: {str(e)}"]
                )
        else:
            baseline_oh = baseline['operations_health']
            baseline_nps = baseline['nps']

        # Run each experiment
        raw_results = []
        for exp_idx, (metric, direction, delta, modified_state) in enumerate(experiment_states, start=1):
            try:
                if batch_results is not None:
                    mod_pred = _get_pred(exp_idx)
                    if mod_pred is None:
                        continue
                    mod_oh = mod_pred['operations_health']
                    mod_nps = mod_pred['nps']
                else:
                    mod_pred = self._predict(modified_state)
                    mod_oh = mod_pred.get('operations_health')
                    mod_nps = mod_pred.get('nps')
                    if mod_oh is None or mod_nps is None:
                        mod_oh = 0.0
                        mod_nps = 0.0
            except Exception as e:
                # Skip this experiment
                continue

            raw = self.analyzer.analyze(
                baseline_state=state,
                modified_state=modified_state,
                delta=delta,
                baseline_oh=baseline_oh,
                baseline_nps=baseline_nps,
                modified_oh=mod_oh,
                modified_nps=mod_nps,
                metric=metric,
                direction=direction
            )
            # Include modified values for aggregation
            raw['modified_oh'] = mod_oh
            raw['modified_nps'] = mod_nps
            raw_results.append(raw)

        if not raw_results:
            return SensitivityResult(
                success=False,
                analyses=[],
                ranking=[],
                errors=["No valid experiment results."]
            )

        # Group by metric and aggregate
        metrics_set = set(r['metric'] for r in raw_results)
        analyses = []
        for metric in metrics_set:
            metric_results = [r for r in raw_results if r['metric'] == metric]
            analysis = self.analyzer.aggregate(
                metric_results,
                metric,
                baseline_oh,
                baseline_nps
            )
            analyses.append(analysis)

        # Rank
        ranked = self.ranker.rank(analyses)

        return SensitivityResult(
            success=True,
            analyses=analyses,
            ranking=ranked,
            warnings=[],
            errors=[],
            metadata={"step_size": self.step_size, "total_experiments": len(experiments)}
        )

    def _predict(self, state: Dict[str, float]) -> Dict[str, Any]:
        """Helper to call PredictionService."""
        op_state = OperationalState.from_dict(state)
        req = PredictionRequest(state=op_state.to_dict(), metadata={"sensitivity": True})
        result = self.service.predict(req)
        return {
            'operations_health': result.operations_health,
            'nps': result.nps,
            'errors': result.errors,
            'warnings': result.warnings
        }
