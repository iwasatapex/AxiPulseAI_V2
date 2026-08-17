"""
ReportEngine – orchestrates report building and export.
"""
import logging
from typing import Dict, Any, Optional
from .models import ReportResult
from .builder import ReportBuilder
from .exporter import ReportExporter

logger = logging.getLogger(__name__)

class ReportEngine:
    def __init__(self):
        self.builder = ReportBuilder()
        self.exporter = ReportExporter()

    def generate(self,
                 forecast_result: Optional[Dict] = None,
                 trend_result: Optional[Any] = None,
                 sensitivity_result: Optional[Any] = None,
                 recommendation_result: Optional[Any] = None,
                 strategy_result: Optional[Any] = None,
                 confidence_result: Optional[Any] = None,
                 risk_result: Optional[Any] = None,
                 explainability_result: Optional[Any] = None,
                 report_type: str = "executive") -> ReportResult:
        try:
            result = self.builder.build(
                forecast_result=forecast_result,
                trend_result=trend_result,
                sensitivity_result=sensitivity_result,
                recommendation_result=recommendation_result,
                strategy_result=strategy_result,
                confidence_result=confidence_result,
                risk_result=risk_result,
                explainability_result=explainability_result,
                report_type=report_type
            )
        except Exception as e:
            logger.exception("Report generation failed")
            return ReportResult(
                success=False,
                title="Report Generation Failed",
                executive_summary=None,
                sections=[],
                appendix=None,
                metadata=None,
                warnings=[],
                errors=[f"Report generation error: {str(e)}"]
            )
        return result

    def export_json(self, result: ReportResult) -> str:
        return self.exporter.to_json(result)

    def export_markdown(self, result: ReportResult) -> str:
        return self.exporter.to_markdown(result)

    def export_text(self, result: ReportResult) -> str:
        return self.exporter.to_text(result)

    def export_dict(self, result: ReportResult) -> Dict[str, Any]:
        return self.exporter.to_dict(result)


# ---------------------------------------------------------------------------
# Module-level compatibility surface
# ---------------------------------------------------------------------------
def generate(*args, **kwargs):
    return ReportEngine().generate(*args, **kwargs)

def export_json(*args, **kwargs):
    return ReportEngine().export_json(*args, **kwargs)

def export_markdown(*args, **kwargs):
    return ReportEngine().export_markdown(*args, **kwargs)

def export_text(*args, **kwargs):
    return ReportEngine().export_text(*args, **kwargs)

def export_dict(*args, **kwargs):
    return ReportEngine().export_dict(*args, **kwargs)
