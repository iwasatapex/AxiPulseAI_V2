"""
SensitivityAnalyzer – computes deltas, sensitivity scores, elasticity.
Supports symmetric experiments and averages results.
"""
from typing import Dict, Any, List, Optional
from ..config import SENSITIVITY_THRESHOLDS

class SensitivityAnalyzer:
    @staticmethod
    def analyze(baseline_state: Dict[str, float],
                modified_state: Dict[str, float],
                delta: float,
                baseline_oh: float,
                baseline_nps: float,
                modified_oh: float,
                modified_nps: float,
                metric: str,
                direction: str = '+') -> Dict[str, Any]:
        """
        Compute sensitivity metrics from a single experiment.
        Returns a dict with raw results.
        """
        oh_change = modified_oh - baseline_oh
        nps_change = modified_nps - baseline_nps

        if abs(delta) < 1e-6:
            sensitivity_oh = 0.0
            sensitivity_nps = 0.0
            elasticity_oh = 0.0
            elasticity_nps = 0.0
        else:
            sensitivity_oh = oh_change / delta
            sensitivity_nps = nps_change / delta
            # Elasticity with safe handling
            input_base = baseline_state.get(metric, 0.0)
            if abs(input_base) < 1e-6:
                elasticity_oh = 0.0
                elasticity_nps = 0.0
            else:
                pct_input = delta / input_base
                if abs(pct_input) < 1e-6:
                    elasticity_oh = 0.0
                    elasticity_nps = 0.0
                else:
                    if abs(baseline_oh) < 1e-6:
                        elasticity_oh = 0.0
                    else:
                        pct_output_oh = oh_change / baseline_oh
                        elasticity_oh = pct_output_oh / pct_input
                    if abs(baseline_nps) < 1e-6:
                        elasticity_nps = 0.0
                    else:
                        pct_output_nps = nps_change / baseline_nps
                        elasticity_nps = pct_output_nps / pct_input

        return {
            'metric': metric,
            'direction': direction,
            'delta': delta,
            'oh_change': oh_change,
            'nps_change': nps_change,
            'sensitivity_oh': sensitivity_oh,
            'sensitivity_nps': sensitivity_nps,
            'elasticity_oh': elasticity_oh,
            'elasticity_nps': elasticity_nps,
        }

    @staticmethod
    def aggregate(results: List[Dict[str, Any]], metric: str,
                  baseline_oh: float, baseline_nps: float) -> 'SensitivityAnalysis':
        """
        Aggregate results from +Δ and −Δ experiments for a single metric.
        Returns a SensitivityAnalysis object.
        """
        from .models import SensitivityAnalysis

        # Separate plus and minus
        plus = [r for r in results if r['direction'] == '+']
        minus = [r for r in results if r['direction'] == '-']

        # Compute average sensitivity
        if plus and minus:
            # Average the absolute sensitivities from both directions
            sens_oh_plus = plus[0]['sensitivity_oh']
            sens_oh_minus = minus[0]['sensitivity_oh']
            # Usually we want signed sensitivity; we can average the signed values
            avg_sens_oh = (sens_oh_plus + sens_oh_minus) / 2.0
            sens_nps_plus = plus[0]['sensitivity_nps']
            sens_nps_minus = minus[0]['sensitivity_nps']
            avg_sens_nps = (sens_nps_plus + sens_nps_minus) / 2.0
            # Elasticity
            el_oh_plus = plus[0]['elasticity_oh']
            el_oh_minus = minus[0]['elasticity_oh']
            avg_el_oh = (el_oh_plus + el_oh_minus) / 2.0
            el_nps_plus = plus[0]['elasticity_nps']
            el_nps_minus = minus[0]['elasticity_nps']
            avg_el_nps = (el_nps_plus + el_nps_minus) / 2.0
            # Use average of absolute changes
            oh_change = (plus[0]['oh_change'] + minus[0]['oh_change']) / 2.0
            nps_change = (plus[0]['nps_change'] + minus[0]['nps_change']) / 2.0
            # Modified outputs: use the plus direction as representative
            modified_oh = plus[0]['modified_oh'] if plus else baseline_oh
            modified_nps = plus[0]['modified_nps'] if plus else baseline_nps
        elif plus:
            # Only plus available
            avg_sens_oh = plus[0]['sensitivity_oh']
            avg_sens_nps = plus[0]['sensitivity_nps']
            avg_el_oh = plus[0]['elasticity_oh']
            avg_el_nps = plus[0]['elasticity_nps']
            oh_change = plus[0]['oh_change']
            nps_change = plus[0]['nps_change']
            modified_oh = plus[0]['modified_oh']
            modified_nps = plus[0]['modified_nps']
        elif minus:
            avg_sens_oh = minus[0]['sensitivity_oh']
            avg_sens_nps = minus[0]['sensitivity_nps']
            avg_el_oh = minus[0]['elasticity_oh']
            avg_el_nps = minus[0]['elasticity_nps']
            oh_change = minus[0]['oh_change']
            nps_change = minus[0]['nps_change']
            modified_oh = minus[0]['modified_oh']
            modified_nps = minus[0]['modified_nps']
        else:
            # No results
            return SensitivityAnalysis(
                metric=metric,
                baseline_output_oh=baseline_oh,
                baseline_output_nps=baseline_nps,
                modified_output_oh=baseline_oh,
                modified_output_nps=baseline_nps,
                operations_health_change=0.0,
                nps_change=0.0,
                sensitivity_score_oh=0.0,
                sensitivity_score_nps=0.0,
                elasticity_oh=0.0,
                elasticity_nps=0.0,
                rank=0,
                classification="Negligible",
                confidence=0.0,
                confidence_type="none",
                metadata={}
            )

        # Classification based on absolute sensitivity score (OH)
        abs_sens = abs(avg_sens_oh)
        thresholds = SENSITIVITY_THRESHOLDS
        if abs_sens > thresholds['very_high']:
            classification = "Very High"
        elif abs_sens > thresholds['high']:
            classification = "High"
        elif abs_sens > thresholds['medium']:
            classification = "Medium"
        elif abs_sens > thresholds['low']:
            classification = "Low"
        else:
            classification = "Negligible"

        # Confidence: derived from consistency between plus and minus. This is
        # a HEURISTIC consistency/floor measure, NOT a statistical confidence.
        # The semantics are surfaced explicitly via confidence_type so the value
        # is never mistaken for a frequentist/Bayesian statistical confidence.
        if plus and minus:
            # Use consistency of sensitivity scores
            diff_oh = abs(plus[0]['sensitivity_oh'] - minus[0]['sensitivity_oh'])
            diff_nps = abs(plus[0]['sensitivity_nps'] - minus[0]['sensitivity_nps'])
            max_sens = max(abs(avg_sens_oh), abs(avg_sens_nps), 1.0)
            consistency = 1.0 - (diff_oh + diff_nps) / (2.0 * max_sens + 1e-6)
            confidence = max(0.3, min(1.0, consistency))
            confidence_type = "heuristic_consistency"
        else:
            confidence = 0.6  # Moderate if only one direction
            confidence_type = "heuristic_single_direction"

        # Build final analysis
        return SensitivityAnalysis(
            metric=metric,
            baseline_output_oh=baseline_oh,
            baseline_output_nps=baseline_nps,
            modified_output_oh=modified_oh,
            modified_output_nps=modified_nps,
            operations_health_change=oh_change,
            nps_change=nps_change,
            sensitivity_score_oh=avg_sens_oh,
            sensitivity_score_nps=avg_sens_nps,
            elasticity_oh=avg_el_oh,
            elasticity_nps=avg_el_nps,
            rank=0,
            classification=classification,
            confidence=confidence,
            confidence_type=confidence_type,
            metadata={"directions": len(results)}
        )
