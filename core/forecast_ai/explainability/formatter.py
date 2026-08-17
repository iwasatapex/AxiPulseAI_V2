import json
from typing import Dict, Any
from .models import ExplainabilityResult, Explanation


def explanation_dict(expl: Explanation):
            return {
                "id": expl.id,
                "title": expl.title,
                "component": expl.component,
                "summary": expl.summary,
                "reasoning": expl.reasoning,
                "conclusion": expl.conclusion,
                "confidence": expl.confidence,
                "source_chain": expl.source_chain,
                "metadata": expl.metadata,
                "evidence": [
                    {
                        "component": e.component,
                        "field": e.field,
                        "value": e.value,
                        "importance": e.importance,
                        "description": e.description,
                        "reference": e.reference,
                        "metadata": e.metadata,
                    }
                    for e in expl.evidence
                ],
            }

class ExplainabilityFormatter:
    @staticmethod
    def _confidence_text(value):
        return f"{value:.2%}" if value is not None else "N/A"

    @staticmethod
    def _explanations(result: ExplainabilityResult):
        return [
            result.forecast_explanation,
            result.trend_explanation,
            result.sensitivity_explanation,
            result.recommendation_explanation,
            result.strategy_explanation,
            result.confidence_explanation,
            result.risk_explanation,
        ]

    @staticmethod
    def to_text(result: ExplainabilityResult) -> str:
        lines = ["Explainability Report", ""]
        lines.append(f"Overall: {result.overall_summary}")
        lines.append("")

        for expl in ExplainabilityFormatter._explanations(result):
            if expl is None:
                continue

            lines.append(
                f"{expl.title} (Confidence: {ExplainabilityFormatter._confidence_text(expl.confidence)})"
            )
            lines.append(f"Summary: {expl.summary}")
            lines.append(f"Reasoning: {expl.reasoning}")
            lines.append(f"Conclusion: {expl.conclusion}")

            if expl.evidence:
                lines.append("Evidence:")
                for ev in expl.evidence:
                    ref = f" [{ev.reference}]" if ev.reference else ""
                    lines.append(
                        f"  - {ev.field}: {ev.value} ({ev.importance}){ref}"
                    )

            lines.append("")

        if result.traces:
            lines.append("Execution Trace:")
            for t in result.traces:
                lines.append(
                    f"  {t.step}. {t.engine}: {t.description}"
                )

        if result.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in result.warnings:
                lines.append(f"  - {w}")

        if result.errors:
            lines.append("")
            lines.append("Errors:")
            for e in result.errors:
                lines.append(f"  - {e}")

        return "\n".join(lines)

    @staticmethod
    def to_markdown(result: ExplainabilityResult) -> str:
        lines = ["# Explainability Report", ""]
        lines.append(f"**Overall:** {result.overall_summary}")
        lines.append("")

        for expl in ExplainabilityFormatter._explanations(result):
            if expl is None:
                continue

            lines.append(f"## {expl.title}")
            lines.append(
                f"**Confidence:** {ExplainabilityFormatter._confidence_text(expl.confidence)}"
            )
            lines.append(f"**Summary:** {expl.summary}")
            lines.append(f"**Reasoning:** {expl.reasoning}")
            lines.append(f"**Conclusion:** {expl.conclusion}")

            if expl.evidence:
                lines.append("**Evidence:**")
                for ev in expl.evidence:
                    ref = f" (`{ev.reference}`)" if ev.reference else ""
                    lines.append(
                        f"- **{ev.field}:** {ev.value} ({ev.importance}){ref}"
                    )

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_dict(result: ExplainabilityResult) -> Dict[str, Any]:

        return {
            "success": result.success,
            "overall_summary": result.overall_summary,
            "forecast": explanation_dict(result.forecast_explanation)
            if result.forecast_explanation
            else None,
            "trend": explanation_dict(result.trend_explanation)
            if result.trend_explanation
            else None,
            "sensitivity": explanation_dict(result.sensitivity_explanation)
            if result.sensitivity_explanation
            else None,
            "recommendation": explanation_dict(result.recommendation_explanation)
            if result.recommendation_explanation
            else None,
            "strategy": explanation_dict(result.strategy_explanation)
            if result.strategy_explanation
            else None,
            "confidence": explanation_dict(result.confidence_explanation)
            if result.confidence_explanation
            else None,
            "risk": explanation_dict(result.risk_explanation)
            if result.risk_explanation
            else None,
            "traces": [
                {
                    "step": t.step,
                    "engine": t.engine,
                    "description": t.description,
                    "purpose": t.purpose,
                    "input_reference": t.input_reference,
                    "output_reference": t.output_reference,
                    "dependencies": t.dependencies,
                    "metadata": t.metadata,
                }
                for t in result.traces
            ],
            "warnings": result.warnings,
            "errors": result.errors,
            "metadata": result.metadata,
        }

    @staticmethod
    def to_json(result: ExplainabilityResult) -> str:
        return json.dumps(
            ExplainabilityFormatter.to_dict(result),
            indent=2,
            default=str,
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
to_text = ExplainabilityFormatter.to_text

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
to_markdown = ExplainabilityFormatter.to_markdown

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
to_dict = ExplainabilityFormatter.to_dict

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
to_json = ExplainabilityFormatter.to_json

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
