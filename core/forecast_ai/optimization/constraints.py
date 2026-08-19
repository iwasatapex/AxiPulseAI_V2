from typing import Dict, Any, List
from .models import Constraint, ConstraintType
from ..config import KPI_BOUNDS

class ConstraintValidator:
    HARD_BOUNDS = KPI_BOUNDS

    @staticmethod
    def validate_hard_bounds(state: Any) -> bool:
        """Canonical hard-bounds check.

        Returns True only when EVERY operational state variable present in
        ``state`` lies within its canonical hard bound (quality 60..100,
        competency 55..100, attendance 65..100, release 50..100,
        transfer 0..20). A state that is outside these bounds is
        operationally invalid and must never be selected as best_solution,
        exposed as a feasible candidate, or used to claim the target was
        achieved.
        """
        if hasattr(state, "to_dict"):
            state = state.to_dict()
        elif not isinstance(state, dict):
            state = vars(state)

        for field, (lo, hi) in ConstraintValidator.HARD_BOUNDS.items():
            if field not in state:
                continue
            try:
                value = float(state[field])
            except (TypeError, ValueError):
                return False
            if not (lo <= value <= hi):
                return False
        return True

    @staticmethod
    def validate(state: Any, constraints: List[Constraint]) -> bool:
        if hasattr(state, "to_dict"):
            state = state.to_dict()
        elif not isinstance(state, dict):
            state = vars(state)

        for c in constraints:
            if c.field not in state:
                continue

            value = state[c.field]

            if c.type == ConstraintType.FIXED:
                if value != c.value:
                    return False
            elif c.type == ConstraintType.MINIMUM:
                if value < c.value:
                    return False
            elif c.type == ConstraintType.MAXIMUM:
                if value > c.value:
                    return False

        return True

    @staticmethod
    def validate_change(original, proposed,
                        constraints: List[Constraint]) -> bool:
        if hasattr(original, "to_dict"):
            original = original.to_dict()
        elif not isinstance(original, dict):
            original = vars(original)

        if hasattr(proposed, "to_dict"):
            proposed = proposed.to_dict()
        elif not isinstance(proposed, dict):
            proposed = vars(proposed)

        for c in constraints:
            if c.type == ConstraintType.MAX_CHANGE:
                if c.field not in original or c.field not in proposed:
                    return False
                change = abs(proposed[c.field] - original[c.field])
                if change > c.value:
                    return False

        return True

validate = ConstraintValidator.validate

validate_change = ConstraintValidator.validate_change

validate_hard_bounds = ConstraintValidator.validate_hard_bounds
