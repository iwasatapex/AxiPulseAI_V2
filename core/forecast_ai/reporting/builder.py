"""
ReportBuilder – assembles report with dynamic ordering, rich executive summary, appendix.
"""
import datetime
import time
from typing import Dict, Any, List, Optional
from .models import ReportSection, ExecutiveSummary, ReportMetadata, ReportResult, ReportAppendix
from .sections import SectionGenerator
from .templates import ReportTemplates

class ReportBuilder:
    @staticmethod
    def build(forecast_result: Dict = None,
              trend_result: Any = None,
              sensitivity_result: Any = None,
              recommendation_result: Any = None,
              strategy_result: Any = None,
              confidence_result: Any = None,
              risk_result: Any = None,
              explainability_result: Any = None,
              report_type: str = "executive") -> ReportResult:
        start_time = time.time()
        # Generate sections with order
        all_sections = []
        generator = SectionGenerator()
        # Map section name to generator method and its result
        section_map = {
            'forecast': (generator.forecast_section, (forecast_result, confidence_result)),
            'trend': (generator.trend_section, (trend_result,)),
            'sensitivity': (generator.sensitivity_section, (sensitivity_result,)),
            'recommendations': (generator.recommendations_section, (recommendation_result,)),
            'strategies': (generator.strategy_section, (strategy_result,)),
            'confidence': (generator.confidence_section, (confidence_result,)),
            'risk': (generator.risk_section, (risk_result,)),
            'explainability': (generator.explainability_section, (explainability_result,)),
        }
        template = ReportTemplates.get_template(report_type)
        ordered_section_names = template.get('sections', ['summary', 'forecast', 'recommendations', 'strategies', 'risks', 'confidence', 'appendix'])
        sections = []
        for name in ordered_section_names:
            if name in section_map:
                method, args = section_map[name]
                try:
                    sec = method(*args)
                    if sec:
                        sections.append(sec)
                except Exception:
                    continue
            elif name == 'summary':
                # summary is handled separately as executive summary
                pass
            elif name == 'appendix':
                # appendix built at end
                pass
        # Sort by order (already set)
        sections.sort(key=lambda s: s.order)
        # Build metadata
        metadata = ReportMetadata(
            generated_at=datetime.datetime.now().isoformat(),
            report_type=report_type,
            version="1.0",
            components=['Components'] + [s.title for s in sections],
            author="ForecastAI",
            forecast_horizon=len(forecast_result.get('timeline', [])) if forecast_result else None,
            execution_duration=time.time() - start_time
        )
        # Build executive summary using all available data
        summary = ReportBuilder._build_executive_summary(
            sections, forecast_result, trend_result, sensitivity_result,
            recommendation_result, strategy_result, confidence_result, risk_result,
            explainability_result
        )
        # Build appendix
        appendix = ReportAppendix(
            metadata={"template": report_type, "component_count": len(sections)},
            warnings=[],
            errors=[],
            components=['Components'] + [s.title for s in sections],
            generation_time=datetime.datetime.now().isoformat()
        )
        # Determine title
        title = template.get('title', 'ForecastAI Report')
        warnings = []
        errors = []
        if not sections:
            warnings.append("No sections generated; some results may be missing.")
        return ReportResult(
            success=len(sections) > 0,
            title=title,
            executive_summary=summary,
            sections=sections,
            appendix=appendix,
            metadata=metadata,
            warnings=warnings,
            errors=errors
        )

    @staticmethod
    def _build_executive_summary(sections: List[ReportSection],
                                 forecast_result: Any,
                                 trend_result: Any,
                                 sensitivity_result: Any,
                                 recommendation_result: Any,
                                 strategy_result: Any,
                                 confidence_result: Any,
                                 risk_result: Any,
                                 explainability_result: Any) -> ExecutiveSummary:
        key_findings = []
        top_recommendations = []
        top_risks = []
        confidence_summary = ""
        operational_outlook = ""
        conclusion = ""

        # Extract findings from forecast
        if forecast_result:
            timeline = forecast_result.get('timeline', [])
            if timeline:
                oh_vals = [d.get('operations_health', 0) for d in timeline if d.get('operations_health') is not None]
                if oh_vals:
                    direction = "increase" if oh_vals[-1] > oh_vals[0] else "decrease" if oh_vals[-1] < oh_vals[0] else "stable"
                    key_findings.append(f"Operations Health forecast: {direction} over {len(timeline)} days.")
        # Trend
        if trend_result and hasattr(trend_result, 'analyses'):
            for a in trend_result.analyses[:2]:
                key_findings.append(f"{a.metric} trend: {a.trend_direction} ({a.pattern})")
        # Sensitivity
        if sensitivity_result and hasattr(sensitivity_result, 'analyses'):
            top_sens = sensitivity_result.analyses[0] if sensitivity_result.analyses else None
            if top_sens:
                key_findings.append(f"Top influencer: {top_sens.metric} (score {top_sens.sensitivity_score_oh:.2f})")
        # Recommendations
        if recommendation_result and hasattr(recommendation_result, 'recommendations'):
            recs = recommendation_result.recommendations
            if recs:
                for r in recs[:3]:
                    top_recommendations.append(f"{r.title} (Priority: {r.priority.value})")
        # Strategies
        if strategy_result and hasattr(strategy_result, 'strategies'):
            strategies = strategy_result.strategies
            if strategies:
                top_recommendations.append(f"Strategy: {strategies[0].name}")
        # Risks
        if risk_result and hasattr(risk_result, 'analyses'):
            for a in risk_result.analyses[:2]:
                if a.overall_risk > 0.3:
                    top_risks.append(f"{a.component} risk: {a.classification}")
                for rf in a.risk_factors[:1]:
                    top_risks.append(f"{rf.name}: {rf.mitigation}")
        # Confidence
        if confidence_result and hasattr(confidence_result, 'overall_confidence'):
            confidence_summary = f"Overall confidence is {confidence_result.overall_confidence:.2%}."
            if confidence_result.overall_confidence > 0.7:
                operational_outlook = "Favorable"
            elif confidence_result.overall_confidence > 0.4:
                operational_outlook = "Cautious"
            else:
                operational_outlook = "Uncertain"
        else:
            confidence_summary = "Confidence data not available."
            operational_outlook = "Unknown"
        # Explainability
        if explainability_result and hasattr(explainability_result, 'overall_summary'):
            conclusion = explainability_result.overall_summary[:200]
        else:
            conclusion = "No explainability summary available."

        overview = "Executive summary generated from all available ForecastAI components."
        return ExecutiveSummary(
            overview=overview,
            key_findings=key_findings[:5],
            top_recommendations=top_recommendations[:5],
            top_risks=top_risks[:5],
            confidence_summary=confidence_summary,
            operational_outlook=operational_outlook,
            conclusion=conclusion
        )

# Module-level compatibility alias
build = ReportBuilder.build
