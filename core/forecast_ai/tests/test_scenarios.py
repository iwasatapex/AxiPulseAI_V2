import unittest
from core.forecast_ai.scenarios import (
    ScenarioManager, ScenarioRegistry, Scenario, Modifier, ModifierType,
    apply_modifiers, merge_modifiers, validate_modifier
)
from core.forecast_ai.state import OperationalState

class TestModifiers(unittest.TestCase):
    def setUp(self):
        self.state = OperationalState(
            quality=85.0,
            competency=78.0,
            transfer=12.0,
            release=92.0,
            attendance=90.0
        )

    def test_add_modifier(self):
        mod = Modifier(field="quality", value=3.0, type=ModifierType.ADD)
        result = apply_modifiers(self.state, [mod])
        self.assertEqual(result.quality, 88.0)

    def test_set_modifier(self):
        mod = Modifier(field="attendance", value=100.0, type=ModifierType.SET)
        result = apply_modifiers(self.state, [mod])
        self.assertEqual(result.attendance, 100.0)

    def test_multiply_modifier(self):
        mod = Modifier(field="transfer", value=1.15, type=ModifierType.MULTIPLY)
        result = apply_modifiers(self.state, [mod])
        self.assertEqual(result.transfer, 13.8)

    def test_validate_modifier(self):
        valid = Modifier(field="quality", value=3.0, type=ModifierType.ADD)
        self.assertTrue(validate_modifier(valid))
        invalid = Modifier(field="", value=3.0, type=ModifierType.ADD)
        self.assertFalse(validate_modifier(invalid))

class TestScenarioRegistry(unittest.TestCase):
    def setUp(self):
        ScenarioRegistry.reset()

    def test_builtin_scenarios_exist(self):
        training = ScenarioRegistry.get("training")
        self.assertIsNotNone(training)
        staff = ScenarioRegistry.get("staff_shortage")
        self.assertIsNotNone(staff)
        high_volume = ScenarioRegistry.get("high_call_volume")
        self.assertIsNotNone(high_volume)
        slowdown = ScenarioRegistry.get("system_slowdown")
        self.assertIsNotNone(slowdown)

    def test_baseline_not_registered(self):
        baseline = ScenarioRegistry.get("baseline")
        self.assertIsNone(baseline)  # Baseline is no-op, not a scenario

    def test_register_custom_scenario(self):
        custom = Scenario(
            id="custom_test",
            name="Custom Test",
            description="Test scenario",
            modifiers=[Modifier(field="quality", value=5.0, type=ModifierType.ADD)]
        )
        ScenarioRegistry.register(custom)
        retrieved = ScenarioRegistry.get("custom_test")
        self.assertEqual(retrieved.id, "custom_test")

class TestScenarioManager(unittest.TestCase):
    def setUp(self):
        ScenarioRegistry.reset()
        self.manager = ScenarioManager()
        self.state = OperationalState(
            quality=85.0,
            competency=78.0,
            transfer=12.0,
            release=92.0,
            attendance=90.0
        )

    def test_baseline_no_change(self):
        result = self.manager.apply_scenarios(self.state, day=1)
        self.assertEqual(result.quality, 85.0)
        self.assertEqual(result.competency, 78.0)

    def test_training_scenario(self):
        result = self.manager.apply_scenarios_to_state(
            self.state,
            scenarios=["training"],
            day=1
        )
        self.assertGreater(result.competency, 78.0)
        self.assertGreater(result.quality, 85.0)

    def test_staff_shortage_scenario(self):
        result = self.manager.apply_scenarios_to_state(
            self.state,
            scenarios=["staff_shortage"],
            day=1
        )
        self.assertLess(result.attendance, 90.0)
        self.assertLess(result.release, 92.0)
        self.assertGreater(result.transfer, 12.0)

    def test_multiple_scenarios(self):
        result = self.manager.apply_scenarios_to_state(
            self.state,
            scenarios=["training", "high_call_volume"],
            day=1
        )
        # Training: +5 competency, +3 quality
        # High Volume: -2 quality, -2 release, +1 transfer
        self.assertGreater(result.competency, 78.0)
        self.assertGreater(result.quality, 85.0)  # net +1 quality (+3 -2)

    def test_priority_ordering(self):
        ScenarioRegistry.register(Scenario(
            id="high_priority",
            name="High Priority",
            description="Should override",
            modifiers=[Modifier(field="quality", value=10.0, type=ModifierType.SET)],
            priority=5
        ))
        ScenarioRegistry.register(Scenario(
            id="low_priority",
            name="Low Priority",
            description="Should be overridden",
            modifiers=[Modifier(field="quality", value=5.0, type=ModifierType.SET)],
            priority=1
        ))
        result = self.manager.apply_scenarios_to_state(
            self.state,
            scenarios=["low_priority", "high_priority"],
            day=1
        )
        self.assertEqual(result.quality, 10.0)

    def test_scheduling(self):
        scheduled = Scenario(
            id="scheduled_test",
            name="Scheduled Test",
            description="Only active on days 2-4",
            modifiers=[Modifier(field="quality", value=5.0, type=ModifierType.ADD)],
            start_day=2,
            end_day=4,
            priority=0
        )
        ScenarioRegistry.register(scheduled)

        result1 = self.manager.apply_scenarios(self.state, day=1)
        self.assertEqual(result1.quality, 85.0)

        result2 = self.manager.apply_scenarios(self.state, day=2)
        self.assertEqual(result2.quality, 90.0)

        result5 = self.manager.apply_scenarios(self.state, day=5)
        self.assertEqual(result5.quality, 85.0)

    def test_immutability(self):
        original = self.state
        self.manager.apply_scenarios(original, day=1)
        self.assertEqual(original.quality, 85.0)  # unchanged

    def test_inactive_scenario(self):
        disabled = Scenario(
            id="disabled_test",
            name="Disabled Test",
            description="Not active",
            modifiers=[Modifier(field="quality", value=10.0, type=ModifierType.ADD)],
            enabled=False
        )
        ScenarioRegistry.register(disabled)
        result = self.manager.apply_scenarios_to_state(
            self.state,
            scenarios=["disabled_test"],
            day=1
        )
        self.assertEqual(result.quality, 85.0)

    def test_validate_scenario(self):
        self.assertTrue(self.manager.validate_scenario("training"))
        self.assertFalse(self.manager.validate_scenario("nonexistent"))

if __name__ == '__main__':
    unittest.main()
