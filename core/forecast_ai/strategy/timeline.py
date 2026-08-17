"""
Timeline generator – creates deterministic milestones from templates.
"""
from typing import List
from .models import Milestone

class TimelineGenerator:
    @staticmethod
    def generate(milestone_defs: List[tuple]) -> List[Milestone]:
        """milestone_defs: list of (week, title, expected_progress)"""
        milestones = []
        for week, title, progress in milestone_defs:
            milestones.append(Milestone(
                week=week,
                title=title,
                description=title,
                expected_progress=progress,
                completed=False
            ))
        return milestones
