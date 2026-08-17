"""
ExplainabilityEngine – orchestrates explanation generation, passing cross‑component results.
"""
import logging
from typing import Dict, Any, Optional
from .models import ExplainabilityResult
from .analyzer import ExplainabilityAnalyzer
from .trace import TraceBuilder

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    def __init__(self):
        self.analyzer = ExplainabilityAnalyzer()
        self.trace_builder = TraceBuilder()

    def explain(self,
                forecast_result: Optional[Dict] = None,
                trend_result: Optional[Any] = None,
                sensitivity_result: Optional[Any] = None,
                recommendation_result: Optional[Any] = None,
                strategy_result: Optional[Any] = None,
                confidence_result: Optional[Any] = None,
                risk_result: Optional[Any] = None) -> ExplainabilityResult:
        """
        Generate explanations for each available component,
        passing cross‑component data to enrich reasoning.
        """
        explanations = {}
        warnings = []
        errors = []

        # Build explanations, passing relevant additional results
        if forecast_result is not None:
            try:
                expl = self.analyzer.analyze_forecast(
                    forecast_result,
                    trend_result=trend_result,
                    sensitivity_result=sensitivity_result,
                    confidence_result=confidence_result,
                    risk_result=risk_result
                )
                if expl:
                    explanations['forecast'] = expl
            except Exception as e:
                errors.append(f"Forecast explanation error: {str(e)}")

        # Other components follow simpler pattern
        for name, result, method in [
            ('trend', trend_result, self.analyzer.analyze_trend),
            ('sensitivity', sensitivity_result, self.analyzer.analyze_sensitivity),
            ('recommendation', recommendation_result, self.analyzer.analyze_recommendations),
            ('strategy', strategy_result, self.analyzer.analyze_strategy),
            ('confidence', confidence_result, self.analyzer.analyze_confidence),
            ('risk', risk_result, self.analyzer.analyze_risk)
        ]:
            if result is not None:
                try:
                    expl = method(result)
                    if expl:
                        explanations[name] = expl
                except Exception as e:
                    errors.append(f"{name} explanation error: {str(e)}")

        if not explanations:
            return ExplainabilityResult(
                success=False,
                overall_summary="No explanations could be generated.",
                warnings=warnings,
                errors=errors or ["No component results provided."]
            )

        # Build traces
        active_components = list(explanations.keys())
        traces = self.trace_builder.build_trace(active_components)

        # Overall summary
        summary_parts = []
        for comp, expl in explanations.items():
            summary_parts.append(f"{comp.title()}: {expl.conclusion}")
        overall_summary = " | ".join(summary_parts)

        result = ExplainabilityResult(
            success=True,
            overall_summary=overall_summary,
            forecast_explanation=explanations.get('forecast'),
            trend_explanation=explanations.get('trend'),
            sensitivity_explanation=explanations.get('sensitivity'),
            recommendation_explanation=explanations.get('recommendation'),
            strategy_explanation=explanations.get('strategy'),
            confidence_explanation=explanations.get('confidence'),
            risk_explanation=explanations.get('risk'),
            traces=traces,
            warnings=warnings,
            errors=errors,
            metadata={"num_explanations": len(explanations)}
        )
        return result

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
explain = ExplainabilityEngine.explain
