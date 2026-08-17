from typing import Dict, Any, List
from .models import Constraint, ConstraintType

class ConstraintValidator:
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
