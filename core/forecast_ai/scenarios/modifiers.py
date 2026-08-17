"""
Modifier functions – pure data transformations on OperationalState.
No forecasting, no prediction, no business logic.
"""
from typing import List
from ..state import OperationalState
from .models import Modifier, ModifierType

def apply_modifier(state: OperationalState, modifier: Modifier) -> OperationalState:
    """Apply a single modifier to an OperationalState. Returns new state."""
    field = modifier.field
    value = modifier.value

    # Get current value from state
    current = getattr(state, field, None)
    if current is None:
        # If field doesn't exist, we could set it, but for known fields we error
        raise ValueError(f"Field '{field}' not found in OperationalState")

    if modifier.type == ModifierType.SET:
        new_val = value
    elif modifier.type == ModifierType.ADD:
        new_val = current + value  # value is absolute points
    elif modifier.type == ModifierType.MULTIPLY:
        new_val = round(current * value, 10)
    else:
        raise ValueError(f"Unknown modifier type: {modifier.type}")

    # Create new state with updated field
    # We use dataclasses.replace for immutability
    from dataclasses import replace
    return replace(state, **{field: new_val})

def apply_modifiers(state: OperationalState, modifiers: List[Modifier]) -> OperationalState:
    """Apply multiple modifiers in order."""
    result = state
    for modifier in modifiers:
        result = apply_modifier(result, modifier)
    return result

def merge_modifiers(modifiers_list: List[List[Modifier]]) -> List[Modifier]:
    """Merge multiple modifier lists, preserving order."""
    merged = []
    for mods in modifiers_list:
        merged.extend(mods)
    return merged

def validate_modifier(modifier: Modifier) -> bool:
    """Validate a modifier's field and value."""
    if not modifier.field or not isinstance(modifier.field, str):
        return False
    if not isinstance(modifier.value, (int, float)):
        return False
    if not isinstance(modifier.type, ModifierType):
        return False
    return True
