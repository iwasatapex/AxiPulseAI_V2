"""
Formatter for Confidence results.
"""
from typing import List, Dict, Any
import json
from .models import ConfidenceResult

def analysis_dict(a):
    return {
        "component": a.component,
        "score": a.confidence_score,
        "classification": a.classification,
        "reasoning": a.reasoning,
        "warnings": a.warnings,
        "metrics": [{"name": m.name, "score": m.score, "weight": m.weight, "reason": m.reason} for m in a.metrics]
    }


class ConfidenceFormatter:
    @staticmethod
    def to_text(result: ConfidenceResult) -> str:
        lines = ["Confidence Report"]
        lines.append(f"Overall Confidence: {result.overall_confidence:.2%}")
        lines.append("")
        for analysis in result.analyses:
            lines.append(f"{analysis.component.title()} Confidence: {analysis.confidence_score:.2%} ({analysis.classification})")
            lines.append(f"  Reasoning: {analysis.reasoning}")
            if analysis.warnings:
                lines.append("  Warnings:")
                for w in analysis.warnings:
                    lines.append(f"    - {w}")
            lines.append("")
        if result.warnings:
            lines.append("Global Warnings:")
            for w in result.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)

    @staticmethod
    def to_markdown(result: ConfidenceResult) -> str:
        lines = ["# Confidence Report\n"]
        lines.append(f"**Overall Confidence:** {result.overall_confidence:.2%}\n")
        for analysis in result.analyses:
            lines.append(f"## {analysis.component.title()} Confidence")
            lines.append(f"- **Score:** {analysis.confidence_score:.2%}")
            lines.append(f"- **Classification:** {analysis.classification}")
            lines.append(f"- **Reasoning:** {analysis.reasoning}")
            if analysis.warnings:
                lines.append("- **Warnings:**")
                for w in analysis.warnings:
                    lines.append(f"  - {w}")
            lines.append("")
        if result.warnings:
            lines.append("## Global Warnings")
            for w in result.warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)

    @staticmethod
    def to_dict(result: ConfidenceResult) -> Dict[str, Any]:
        def analysis_dict(a):
            return {
                "component": a.component,
                "score": a.confidence_score,
                "classification": a.classification,
                "reasoning": a.reasoning,
                "warnings": a.warnings,
                "metrics": [{"name": m.name, "score": m.score, "weight": m.weight, "reason": m.reason} for m in a.metrics]
            }
        return {
            "success": result.success,
            "overall_confidence": result.overall_confidence,
            "analyses": [analysis_dict(a) for a in result.analyses],
            "warnings": result.warnings,
            "errors": result.errors,
            "metadata": result.metadata
        }

    @staticmethod
    def to_json(result: ConfidenceResult) -> str:
        return json.dumps(ConfidenceFormatter.to_dict(result), indent=2, default=str)

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
to_text = ConfidenceFormatter.to_text

# Module-level compatibility surface.
# Delegates to existing implementations; no logic changed.
to_markdown = ConfidenceFormatter.to_markdown
to_dict = ConfidenceFormatter.to_dict
to_json = ConfidenceFormatter.to_json
