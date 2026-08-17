from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

class ModifierType(Enum):
    SET = "set"           # Replace value entirely
    ADD = "add"           # Add absolute points (KPI scale is 0-100)
    MULTIPLY = "multiply" # Multiply existing value

@dataclass
class Modifier:
    """A single modifier to apply to a state field."""
    field: str
    value: float
    type: ModifierType = ModifierType.ADD

@dataclass
class Scenario:
    """A scenario with modifiers and scheduling."""
    id: str
    name: str
    description: str
    modifiers: List[Modifier]
    enabled: bool = True
    priority: int = 0  # Higher = applied later (overrides)
    start_day: Optional[int] = None  # None = from day 1
    end_day: Optional[int] = None    # None = until end
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, day: int) -> bool:
        if not self.enabled:
            return False
        if self.start_day is not None and day < self.start_day:
            return False
        if self.end_day is not None and day > self.end_day:
            return False
        return True
