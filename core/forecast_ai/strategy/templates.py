"""
Strategy templates – deterministic, template-driven strategy generation.
"""
from typing import Dict, Any, List, Tuple
from .models import StrategyCategory

class StrategyTemplates:
    @staticmethod
    def get_template(category: StrategyCategory) -> Dict[str, Any]:
        templates = {
            StrategyCategory.OPERATIONAL_EXCELLENCE: {
                "name": "Operational Excellence Program",
                "description": "Comprehensive operational improvement across all KPIs.",
                "objective": "Achieve balanced operational excellence.",
                "default_risks": [
                    "Temporary productivity dip during implementation",
                    "Resistance to change from staff",
                    "Resource allocation conflicts"
                ],
                "default_milestones": [
                    (1, "Launch operational review", 0.1),
                    (2, "Implement quality improvements", 0.3),
                    (3, "Roll out training programs", 0.6),
                    (4, "Evaluate operational metrics", 0.8),
                    (6, "Full operational excellence assessment", 1.0)
                ],
                "estimated_duration_weeks": 6,
                "estimated_complexity": 0.7,
                "estimated_disruption": 0.5
            },
            StrategyCategory.TRAINING: {
                "name": "Training Excellence Strategy",
                "description": "Focus on competency and quality through training.",
                "objective": "Elevate agent competency and quality scores.",
                "default_risks": [
                    "Training budget constraints",
                    "Scheduling conflicts with operations",
                    "Training effectiveness lag"
                ],
                "default_milestones": [
                    (1, "Launch training needs assessment", 0.2),
                    (2, "Begin core training modules", 0.4),
                    (3, "Implement coaching sessions", 0.6),
                    (4, "Evaluate training impact", 0.8),
                    (5, "Continuous improvement cycle", 1.0)
                ],
                "estimated_duration_weeks": 5,
                "estimated_complexity": 0.4,
                "estimated_disruption": 0.2
            },
            StrategyCategory.CUSTOMER_EXPERIENCE: {
                "name": "Customer Experience Strategy",
                "description": "Enhance customer satisfaction through service improvements.",
                "objective": "Improve NPS and customer retention.",
                "default_risks": [
                    "Customer feedback delays",
                    "Changes in customer expectations",
                    "Operational capacity constraints"
                ],
                "default_milestones": [
                    (1, "Analyze customer feedback", 0.1),
                    (2, "Improve first-contact resolution", 0.3),
                    (3, "Reduce call transfers", 0.5),
                    (4, "Enhance service quality", 0.7),
                    (6, "Customer satisfaction review", 1.0)
                ],
                "estimated_duration_weeks": 6,
                "estimated_complexity": 0.6,
                "estimated_disruption": 0.4
            },
            StrategyCategory.BALANCED: {
                "name": "Balanced Improvement Plan",
                "description": "Gradual, balanced improvements across operations.",
                "objective": "Sustain operational stability with incremental gains.",
                "default_risks": [
                    "Slower-than-expected progress",
                    "Resource competition between initiatives",
                    "Measurement delays"
                ],
                "default_milestones": [
                    (1, "Kickoff balanced improvement", 0.1),
                    (2, "Implement quality and competency", 0.3),
                    (4, "Address attendance and transfer", 0.6),
                    (6, "Monitor operational health", 0.8),
                    (8, "Balanced operations review", 1.0)
                ],
                "estimated_duration_weeks": 8,
                "estimated_complexity": 0.5,
                "estimated_disruption": 0.3
            },
            StrategyCategory.STAFFING: {
                "name": "Workforce Stability Strategy",
                "description": "Strengthen attendance and staffing reliability.",
                "objective": "Reduce absenteeism and improve schedule adherence.",
                "default_risks": [
                    "Hiring delays",
                    "Unforeseen staff turnover",
                    "Seasonal variations"
                ],
                "default_milestones": [
                    (1, "Review workforce policies", 0.2),
                    (2, "Implement attendance improvements", 0.4),
                    (3, "Enhance scheduling", 0.6),
                    (4, "Evaluate staffing efficiency", 0.8),
                    (5, "Workforce stability review", 1.0)
                ],
                "estimated_duration_weeks": 5,
                "estimated_complexity": 0.4,
                "estimated_disruption": 0.3
            }
        }
        return templates.get(category, templates[StrategyCategory.BALANCED])

    @staticmethod
    def get_priority(score: float) -> str:
        """Map optimization score to strategy priority."""
        if score < 0.5:
            return "Critical"
        elif score < 1.0:
            return "High"
        elif score < 2.0:
            return "Medium"
        else:
            return "Low"
