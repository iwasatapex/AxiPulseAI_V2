"""
Explanation templates – structured parts for each component.
"""
from typing import Dict, Any, List

class ExplanationTemplates:
    @staticmethod
    def get_template(component: str) -> Dict[str, Any]:
        templates = {
            'forecast': {
                'title': 'Forecast Explanation',
                'summary_template': 'The forecast predicts {direction} over the next {horizon} days for Operations Health and NPS based on the current operational state and scenario assumptions.',
                'reasoning_template': 'The forecast indicates {direction} because {drivers}. {additional}',
                'conclusion_template': 'Operations Health is expected to {direction} to {oh_value}, with NPS reaching {nps_value}.',
                'warnings_template': 'Consider the following: {warnings}',
                'default_drivers': ['historical trends', 'current operational metrics']
            },
            'trend': {
                'title': 'Trend Explanation',
                'summary_template': 'Trend analysis reveals {direction} of {metric} with {strength} strength and {volatility} volatility.',
                'reasoning_template': 'The {metric} trend shows {direction} because {evidence}. The pattern is {pattern}.',
                'conclusion_template': '{metric} is {direction} with {strength} strength ({pattern}).',
                'warnings_template': 'Volatility is {volatility}; monitor closely.'
            },
            'sensitivity': {
                'title': 'Sensitivity Explanation',
                'summary_template': 'Sensitivity analysis measures the influence of each KPI on Operations Health and NPS.',
                'reasoning_template': '{metric} has {classification} influence (score {score:.2f}) on Operations Health because {reason}.',
                'conclusion_template': 'Top influencer: {metric} (score {score:.2f}).',
                'warnings_template': 'Weak signals for {weak_metrics} indicate limited leverage.'
            },
            'recommendation': {
                'title': 'Recommendation Explanation',
                'summary_template': '{count} recommendations have been generated to improve operational performance.',
                'reasoning_template': 'Recommendations are derived from optimization results; top priority is {priority}. They target {areas}.',
                'conclusion_template': 'Focus on {top_actions} for immediate impact.',
                'warnings_template': '{overload} may hinder execution; consider prioritization.'
            },
            'strategy': {
                'title': 'Strategy Explanation',
                'summary_template': '{count} strategies have been developed, with "{best}" ranked highest.',
                'reasoning_template': 'Strategies group recommendations into coherent plans. "{best}" scores {score:.2f} due to {reasons}.',
                'conclusion_template': 'Adopt "{best}" to achieve balanced operational improvement.',
                'warnings_template': 'Risks identified: {risks}.'
            },
            'confidence': {
                'title': 'Confidence Explanation',
                'summary_template': 'Overall confidence is {score:.2%} ({classification}).',
                'reasoning_template': 'Confidence is driven by {drivers}. {details}',
                'conclusion_template': 'Confidence level: {classification} ({score:.2%}).',
                'warnings_template': 'Low confidence components: {low_components}.'
            },
            'risk': {
                'title': 'Risk Explanation',
                'summary_template': 'Overall risk is {score:.2%} ({classification}).',
                'reasoning_template': 'Top risks: {top_risks}. {details}',
                'conclusion_template': 'Risk classification: {classification}.',
                'warnings_template': 'Critical risks: {critical_risks}.'
            }
        }
        return templates.get(component, {
            'title': 'Explanation',
            'summary_template': 'Analysis complete.',
            'reasoning_template': 'No detailed reasoning available.',
            'conclusion_template': 'No conclusion available.',
            'warnings_template': ''
        })

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
get_template = ExplanationTemplates.get_template
