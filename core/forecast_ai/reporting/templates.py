"""
Report templates – each defines sections, ordering, and style.
"""
from typing import Dict, Any, List

class ReportTemplates:
    @staticmethod
    def get_template(report_type: str) -> Dict[str, Any]:
        templates = {
            'executive': {
                'title': 'Executive Operations Report',
                'description': 'High-level operational overview for decision makers.',
                'sections': ['summary', 'forecast', 'recommendations', 'strategies', 'risks', 'confidence', 'explainability', 'appendix'],
                'executive_style': 'concise',
                'detail_level': 'high',
            },
            'operational': {
                'title': 'Operational Performance Report',
                'description': 'Detailed operational metrics and analysis.',
                'sections': ['summary', 'forecast', 'trend', 'sensitivity', 'recommendations', 'strategies', 'confidence', 'risk', 'explainability', 'appendix'],
                'executive_style': 'detailed',
                'detail_level': 'full',
            },
            'technical': {
                'title': 'Technical ForecastAI Report',
                'description': 'Comprehensive technical output from all engines.',
                'sections': ['summary', 'forecast', 'trend', 'sensitivity', 'recommendations', 'strategies', 'confidence', 'risk', 'explainability', 'appendix'],
                'executive_style': 'technical',
                'detail_level': 'full',
            },
            'management': {
                'title': 'Management Summary',
                'description': 'Management-focused summary of key insights.',
                'sections': ['summary', 'forecast', 'recommendations', 'strategies', 'risks', 'confidence', 'appendix'],
                'executive_style': 'concise',
                'detail_level': 'medium',
            },
            'audit': {
                'title': 'ForecastAI Audit Report',
                'description': 'Detailed audit of forecasts, recommendations, and strategies.',
                'sections': ['summary', 'forecast', 'trend', 'sensitivity', 'recommendations', 'strategies', 'confidence', 'risk', 'explainability', 'appendix'],
                'executive_style': 'detailed',
                'detail_level': 'full',
            },
            'dashboard': {
                'title': 'ForecastAI Dashboard Payload',
                'description': 'API-ready payload for dashboard integration.',
                'sections': ['summary', 'forecast', 'recommendations', 'strategies', 'risks', 'confidence'],
                'executive_style': 'concise',
                'detail_level': 'low',
            },
            'json': {
                'title': 'ForecastAI JSON Export',
                'description': 'Structured JSON export of all results.',
                'sections': ['summary', 'forecast', 'trend', 'sensitivity', 'recommendations', 'strategies', 'confidence', 'risk', 'explainability'],
                'executive_style': 'technical',
                'detail_level': 'full',
            },
            'markdown': {
                'title': 'ForecastAI Markdown Report',
                'description': 'Human-readable markdown report.',
                'sections': ['summary', 'forecast', 'trend', 'sensitivity', 'recommendations', 'strategies', 'confidence', 'risk', 'explainability', 'appendix'],
                'executive_style': 'detailed',
                'detail_level': 'full',
            }
        }
        return templates.get(report_type, templates['operational'])


# ---------------------------------------------------------------------------
# Module-level compatibility surface
# ---------------------------------------------------------------------------
def get_template(*args, **kwargs):
    return ReportTemplates.get_template(*args, **kwargs)
