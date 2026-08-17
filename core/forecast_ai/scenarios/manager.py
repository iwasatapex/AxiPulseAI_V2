"""
ScenarioManager – Applies scenarios to OperationalState.
Pure orchestration – no prediction, no evolution.
"""
from typing import List, Optional
from .models import Scenario
from .registry import ScenarioRegistry
from .modifiers import apply_modifiers
from ..state import OperationalState

class ScenarioManager:
    def __init__(self):
        self.registry = ScenarioRegistry

    def get_active_scenarios(self, day: int) -> List[Scenario]:
        return self.registry.get_active(day)

    def apply_scenarios(self, state: OperationalState, day: int) -> OperationalState:
        """
        Apply all scheduled active scenarios.
        """
        active = self.get_active_scenarios(day)
        if not active:
            return state

        all_modifiers = []
        for scenario in active:
            all_modifiers.extend(scenario.modifiers)

        return apply_modifiers(state, all_modifiers)

    def apply_scenarios_to_state(self, state: OperationalState, scenarios: List[str], day: int) -> OperationalState:
        """
        Apply specific scenarios by ID.
        """
        selected = []
        for sid in scenarios:
            scenario = self.registry.get(sid)
            if scenario and scenario.is_active(day):
                selected.append(scenario)

        if not selected:
            return state

        all_modifiers = []
        for scenario in selected:
            all_modifiers.extend(scenario.modifiers)
        return apply_modifiers(state, all_modifiers)

    def validate_scenario(self, scenario_id: str) -> bool:
        scenario = self.registry.get(scenario_id)
        return scenario is not None and scenario.enabled
