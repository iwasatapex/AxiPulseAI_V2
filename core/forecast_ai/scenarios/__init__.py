"""Scenario management for ForecastAI – "What if?" analysis."""
from .models import Scenario, Modifier, ModifierType
from .manager import ScenarioManager
from .registry import ScenarioRegistry
from .modifiers import apply_modifiers, merge_modifiers, validate_modifier
