"""
Scenario registry – stores and provides built-in scenarios.
"""
from typing import Dict, Optional, List
from .models import Scenario, Modifier, ModifierType

class ScenarioRegistry:
    _scenarios: Dict[str, Scenario] = {}

    @classmethod
    def register(cls, scenario: Scenario) -> None:
        cls._scenarios[scenario.id] = scenario

    @classmethod
    def get(cls, scenario_id: str) -> Optional[Scenario]:
        return cls._scenarios.get(scenario_id)

    @classmethod
    def list(cls) -> List[Scenario]:
        return list(cls._scenarios.values())

    @classmethod
    def get_active(cls, day: int) -> List[Scenario]:
        """Return only scheduled active scenarios."""
        builtin = {
            "training",
            "staff_shortage",
            "high_call_volume",
            "system_slowdown",
            "baseline",
        }

        active = [
            s for s in cls._scenarios.values()
            if s.id not in builtin and s.is_active(day)
        ]
        return sorted(active, key=lambda s: s.priority)

    @classmethod
    def reset(cls) -> None:
        cls._scenarios.clear()
        cls._register_builtins()

    @classmethod
    def _register_builtins(cls) -> None:
        # Baseline is NOT registered – it's the default no-op.
        # Training – absolute points (+5 competency, +3 quality)
        cls.register(Scenario(
            id="training",
            name="Training Improvement",
            description="Increased competency and quality through training.",
            modifiers=[
                Modifier(field="competency", value=5.0, type=ModifierType.ADD),
                Modifier(field="quality", value=3.0, type=ModifierType.ADD)
            ],
            priority=1,
            enabled=True
        ))
        # Staff Shortage – absolute points (-5 attendance, -3 release, +2 transfer)
        cls.register(Scenario(
            id="staff_shortage",
            name="Staff Shortage",
            description="Reduced attendance, lower release, higher transfers.",
            modifiers=[
                Modifier(field="attendance", value=-5.0, type=ModifierType.ADD),
                Modifier(field="release", value=-3.0, type=ModifierType.ADD),
                Modifier(field="transfer", value=2.0, type=ModifierType.ADD)
            ],
            priority=2,
            enabled=True
        ))
        # High Call Volume – absolute points (-2 quality, -2 release, +1 transfer)
        cls.register(Scenario(
            id="high_call_volume",
            name="High Call Volume",
            description="Increased call volume leads to quality drop and transfers.",
            modifiers=[
                Modifier(field="quality", value=-2.0, type=ModifierType.ADD),
                Modifier(field="release", value=-2.0, type=ModifierType.ADD),
                Modifier(field="transfer", value=1.0, type=ModifierType.ADD)
            ],
            priority=2,
            enabled=True
        ))
        # System Slowdown – absolute points (-3 competency, -2 quality, -2 release)
        cls.register(Scenario(
            id="system_slowdown",
            name="System Slowdown",
            description="System performance issues reduce competency and quality.",
            modifiers=[
                Modifier(field="competency", value=-3.0, type=ModifierType.ADD),
                Modifier(field="quality", value=-2.0, type=ModifierType.ADD),
                Modifier(field="release", value=-2.0, type=ModifierType.ADD)
            ],
            priority=3,
            enabled=True
        ))

ScenarioRegistry.reset()
