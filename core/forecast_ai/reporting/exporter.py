"""
ReportExporter – exports ReportResult to text, markdown, json, dict with shared helpers.
"""
import json
from typing import Dict, Any
from .models import ReportResult

class ReportExporter:
    @staticmethod
    def _render_summary(exec_summary) -> str:
        lines = []
        lines.append("## Executive Summary")
        lines.append(exec_summary.overview)
        if exec_summary.key_findings:
            lines.append("**Key Findings:**")
            for f in exec_summary.key_findings:
                lines.append(f"- {f}")
        if exec_summary.top_recommendations:
            lines.append("**Top Recommendations:**")
            for r in exec_summary.top_recommendations:
                lines.append(f"- {r}")
        if exec_summary.top_risks:
            lines.append("**Top Risks:**")
            for r in exec_summary.top_risks:
                lines.append(f"- {r}")
        lines.append(f"**Confidence:** {exec_summary.confidence_summary}")
        lines.append(f"**Outlook:** {exec_summary.operational_outlook}")
        lines.append(f"**Conclusion:** {exec_summary.conclusion}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_sections(sections) -> str:
        lines = ["## Detailed Sections"]
        for sec in sections:
            lines.append(f"### {sec.title}")
            lines.append(sec.content)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_appendix(appendix, warnings, errors) -> str:
        lines = ["## Appendix"]
        lines.append(f"**Generated:** {appendix.generation_time}")
        lines.append(f"**Components:** {', '.join(appendix.components)}")
        if warnings:
            lines.append("**Warnings:**")
            for w in warnings:
                lines.append(f"- {w}")
        if errors:
            lines.append("**Errors:**")
            for e in errors:
                lines.append(f"- {e}")
        return "\n".join(lines)

    @staticmethod
    def to_dict(result: ReportResult) -> Dict[str, Any]:
        return {
            "success": result.success,
            "title": result.title,
            "executive_summary": {
                "overview": result.executive_summary.overview,
                "key_findings": result.executive_summary.key_findings,
                "top_recommendations": result.executive_summary.top_recommendations,
                "top_risks": result.executive_summary.top_risks,
                "confidence_summary": result.executive_summary.confidence_summary,
                "operational_outlook": result.executive_summary.operational_outlook,
                "conclusion": result.executive_summary.conclusion
            },
            "sections": [{"title": s.title, "content": s.content, "order": s.order} for s in result.sections],
            "appendix": {
                "metadata": result.appendix.metadata,
                "warnings": result.appendix.warnings,
                "errors": result.appendix.errors,
                "components": result.appendix.components,
                "generation_time": result.appendix.generation_time
            },
            "metadata": {
                "generated_at": result.metadata.generated_at,
                "report_type": result.metadata.report_type,
                "version": result.metadata.version,
                "components": result.metadata.components,
                "forecast_horizon": result.metadata.forecast_horizon,
                "execution_duration": result.metadata.execution_duration
            },
            "warnings": result.warnings,
            "errors": result.errors
        }

    @staticmethod
    def to_json(result: ReportResult) -> str:
        return json.dumps(ReportExporter.to_dict(result), indent=2, default=str)

    @staticmethod
    def to_text(result: ReportResult) -> str:
        lines = [f"# {result.title}\n"]
        if result.executive_summary:
            lines.append(ReportExporter._render_summary(result.executive_summary))
        if result.sections:
            lines.append(ReportExporter._render_sections(result.sections))
        if result.appendix:
            lines.append(ReportExporter._render_appendix(result.appendix, result.warnings, result.errors))
        return "\n".join(lines)

    @staticmethod
    def to_markdown(result: ReportResult) -> str:
        # Similar to text but markdown-friendly
        lines = [f"# {result.title}\n"]
        if result.executive_summary:
            lines.append(ReportExporter._render_summary(result.executive_summary))
        if result.sections:
            lines.append(ReportExporter._render_sections(result.sections))
        if result.appendix:
            lines.append(ReportExporter._render_appendix(result.appendix, result.warnings, result.errors))
        return "\n".join(lines)

    @staticmethod
    def to_text_simple(result: ReportResult) -> str:
        return ReportExporter.to_text(result)


# ---------------------------------------------------------------------------
# Module-level compatibility surface
# ---------------------------------------------------------------------------
def to_dict(*args, **kwargs):
    return ReportExporter.to_dict(*args, **kwargs)

def to_json(*args, **kwargs):
    return ReportExporter.to_json(*args, **kwargs)

def to_text(*args, **kwargs):
    return ReportExporter.to_text(*args, **kwargs)

def to_markdown(*args, **kwargs):
    return ReportExporter.to_markdown(*args, **kwargs)

def to_text_simple(*args, **kwargs):
    return ReportExporter.to_text_simple(*args, **kwargs)
