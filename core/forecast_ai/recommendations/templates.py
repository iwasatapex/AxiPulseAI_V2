"""
Templates mapping operational changes to recommendation text.
"""
from typing import Dict, Tuple, List
from .models import Category, Difficulty, Priority

class RecommendationTemplates:
    @staticmethod
    def get_template(field: str) -> Tuple[str, str, Category]:
        """Return (title, description, category) for a field change."""
        templates = {
            'quality': (
                "Improve Quality Assurance",
                "Enhance quality monitoring and feedback processes to increase service quality.",
                Category.QUALITY
            ),
            'competency': (
                "Increase Coaching and Training",
                "Boost agent competency through targeted training programs.",
                Category.COMPETENCY
            ),
            'attendance': (
                "Improve Workforce Adherence",
                "Enhance schedule adherence and reduce absenteeism.",
                Category.ATTENDANCE
            ),
            'transfer': (
                "Reduce Unnecessary Call Transfers",
                "Empower agents to resolve issues without transferring.",
                Category.TRANSFER
            ),
            'release': (
                "Improve First-Contact Resolution",
                "Increase release rates by addressing issues during the first call.",
                Category.RELEASE
            )
        }
        return templates.get(field, ("Adjust Operations", "Review operational practices.", Category.GENERAL))

    @staticmethod
    def get_actions(field: str) -> List[str]:
        actions_map = {
            'quality': ["Implement QA scorecards", "Conduct calibration sessions", "Increase monitoring frequency"],
            'competency': ["Launch training modules", "Schedule coaching sessions", "Create upskilling plans"],
            'attendance': ["Reinforce attendance policies", "Offer flexible scheduling", "Improve workforce planning"],
            'transfer': ["Review transfer protocols", "Empower agents with decision rights", "Enhance knowledge base"],
            'release': ["Set FCR targets", "Analyze release bottlenecks", "Provide resolution training"]
        }
        return actions_map.get(field, ["Review operational process", "Implement best practices"])

    @staticmethod
    def get_difficulty(change_magnitude: float) -> Difficulty:
        """Map change magnitude to difficulty."""
        if abs(change_magnitude) < 0.5:
            return Difficulty.VERY_EASY
        elif abs(change_magnitude) < 1.5:
            return Difficulty.EASY
        elif abs(change_magnitude) < 3.0:
            return Difficulty.MEDIUM
        elif abs(change_magnitude) < 5.0:
            return Difficulty.HARD
        else:
            return Difficulty.VERY_HARD

get_template = RecommendationTemplates.get_template

get_actions = RecommendationTemplates.get_actions

get_difficulty = RecommendationTemplates.get_difficulty
