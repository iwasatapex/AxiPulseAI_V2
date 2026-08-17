"""
SectionGenerator – builds rich report sections from component results.
"""
import datetime
from typing import Dict, Any, List, Optional
from .models import ReportSection, ReportAppendix

class SectionGenerator:
    @staticmethod
    def forecast_section(forecast_result: Dict, confidence_result: Any = None) -> Optional[ReportSection]:
        if not forecast_result:
            return None
        timeline = forecast_result.get('timeline', [])
        if not timeline:
            return None
        oh_vals = [d.get('operations_health', 0) for d in timeline if d.get('operations_health') is not None]
        nps_vals = [d.get('nps', 0) for d in timeline if d.get('nps') is not None]
        if not oh_vals:
            return None
        horizon = len(timeline)
        avg_oh = sum(oh_vals)/len(oh_vals)
        min_oh = min(oh_vals)
        max_oh = max(oh_vals)
        change_oh = oh_vals[-1] - oh_vals[0]
        direction = "increase" if change_oh > 0 else "decrease" if change_oh < 0 else "stable"
        avg_nps = sum(nps_vals)/len(nps_vals) if nps_vals else None
        # confidence
        confidence_val = None
        if confidence_result and hasattr(confidence_result, 'overall_confidence'):
            confidence_val = confidence_result.overall_confidence
        content_lines = [
            f"Horizon: {horizon} days",
            f"Average OH: {avg_oh:.1f}",
            f"OH range: {min_oh:.1f} – {max_oh:.1f}",
            f"Net OH change: {change_oh:+.1f} ({direction})"
        ]
        if avg_nps is not None:
            content_lines.append(f"Average NPS: {avg_nps:.1f}")
        if confidence_val is not None:
            content_lines.append(f"Confidence: {confidence_val:.0%}")
        content = "\n".join(content_lines)
        return ReportSection(title="Forecast Summary", content=content, order=10)

    @staticmethod
    def trend_section(trend_result: Any) -> Optional[ReportSection]:
        if not trend_result or not hasattr(trend_result, 'analyses'):
            return None
        analyses = trend_result.analyses
        if not analyses:
            return None
        lines = ["Trend Analysis:"]
        for a in analyses[:5]:
            lines.append(f"  {a.metric}: {a.trend_direction} ({a.pattern}), strength {a.trend_strength}, volatility {a.volatility}")
        content = "\n".join(lines)
        return ReportSection(title="Trend Intelligence", content=content, order=20)

    @staticmethod
    def sensitivity_section(sensitivity_result: Any) -> Optional[ReportSection]:
        if not sensitivity_result or not hasattr(sensitivity_result, 'analyses'):
            return None
        analyses = sensitivity_result.analyses
        if not analyses:
            return None
        lines = ["Sensitivity Ranking:"]
        for a in analyses[:5]:
            lines.append(f"  {a.metric}: OH influence {a.sensitivity_score_oh:.3f}, NPS {a.sensitivity_score_nps:.3f}")
        content = "\n".join(lines)
        return ReportSection(title="KPI Sensitivity", content=content, order=30)

    @staticmethod
    def recommendations_section(rec_result: Any) -> Optional[ReportSection]:
        if not rec_result or not hasattr(rec_result, 'recommendations'):
            return None
        recs = rec_result.recommendations
        if not recs:
            return None
        lines = [f"Top {min(5, len(recs))} Recommendations:"]
        for i, r in enumerate(recs[:5], 1):
            lines.append(f"  {i}. {r.title} (Priority: {r.priority.value}, Difficulty: {r.difficulty.value})")
            if r.estimated_operations_health_gain:
                lines.append(f"     Expected OH gain: {r.estimated_operations_health_gain:.2f}")
        content = "\n".join(lines)
        return ReportSection(title="Key Recommendations", content=content, order=40)

    @staticmethod
    def strategy_section(strategy_result: Any) -> Optional[ReportSection]:
        if not strategy_result or not hasattr(strategy_result, 'strategies'):
            return None
        strategies = strategy_result.strategies
        if not strategies:
            return None
        lines = [f"Top Strategy: {strategies[0].name} (Priority: {strategies[0].priority})"]
        if len(strategies) > 1:
            lines.append(f"Alternate: {', '.join([s.name for s in strategies[1:3]])}")
        for s in strategies[:1]:
            lines.append(f"  Duration: {s.estimated_duration_weeks} weeks, Complexity: {s.estimated_complexity:.2f}, Risk: {s.estimated_disruption:.2f}")
        content = "\n".join(lines)
        return ReportSection(title="Strategies", content=content, order=50)

    @staticmethod
    def confidence_section(confidence_result: Any) -> Optional[ReportSection]:
        if not confidence_result or not hasattr(confidence_result, 'overall_confidence'):
            return None
        lines = [f"Overall Confidence: {confidence_result.overall_confidence:.0%}"]
        if hasattr(confidence_result, 'analyses'):
            for a in confidence_result.analyses[:5]:
                lines.append(f"  {a.component}: {a.confidence_score:.2%} ({a.classification})")
        content = "\n".join(lines)
        return ReportSection(title="Confidence Assessment", content=content, order=60)

    @staticmethod
    def risk_section(risk_result: Any) -> Optional[ReportSection]:
        if not risk_result or not hasattr(risk_result, 'overall_risk'):
            return None
        lines = [f"Overall Risk: {risk_result.overall_risk:.2%}"]
        if hasattr(risk_result, 'analyses'):
            for a in risk_result.analyses[:5]:
                lines.append(f"  {a.component}: {a.overall_risk:.2%} ({a.classification})")
                for rf in a.risk_factors[:2]:
                    lines.append(f"    - {rf.name}: {rf.risk_score:.2f} (Mitigation: {rf.mitigation})")
        content = "\n".join(lines)
        return ReportSection(title="Risk Assessment", content=content, order=70)

    @staticmethod
    def explainability_section(explainability_result: Any) -> Optional[ReportSection]:
        if not explainability_result or not hasattr(explainability_result, 'overall_summary'):
            return None
        content = explainability_result.overall_summary
        if hasattr(explainability_result, 'traces'):
            content += f"\n\nExecution Trace: {len(explainability_result.traces)} steps"
        return ReportSection(title="Explainability", content=content, order=80)

    @staticmethod
    def appendix_section(result_metadata: Dict[str, Any], warnings: List[str], errors: List[str],
                         components: List[str]) -> Optional[ReportSection]:
        lines = ["Appendix:"]
        lines.append(f"  Generated: {datetime.datetime.now().isoformat()}")
        if components:
            lines.append(f"  Components: {', '.join(components)}")
        if warnings:
            lines.append(f"  Warnings: {len(warnings)}")
        if errors:
            lines.append(f"  Errors: {len(errors)}")
        content = "\n".join(lines)
        return ReportSection(title="Appendix", content=content, order=999)


# ---------------------------------------------------------------------------
# Module-level compatibility surface
# ---------------------------------------------------------------------------
def forecast_section(*args, **kwargs):
    return SectionGenerator.forecast_section(*args, **kwargs)

def trend_section(*args, **kwargs):
    return SectionGenerator.trend_section(*args, **kwargs)

def sensitivity_section(*args, **kwargs):
    return SectionGenerator.sensitivity_section(*args, **kwargs)

def recommendations_section(*args, **kwargs):
    return SectionGenerator.recommendations_section(*args, **kwargs)

def strategy_section(*args, **kwargs):
    return SectionGenerator.strategy_section(*args, **kwargs)

def confidence_section(*args, **kwargs):
    return SectionGenerator.confidence_section(*args, **kwargs)

def risk_section(*args, **kwargs):
    return SectionGenerator.risk_section(*args, **kwargs)

def explainability_section(*args, **kwargs):
    return SectionGenerator.explainability_section(*args, **kwargs)

def appendix_section(*args, **kwargs):
    return SectionGenerator.appendix_section(*args, **kwargs)
