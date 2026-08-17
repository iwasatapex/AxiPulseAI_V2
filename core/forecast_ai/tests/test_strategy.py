import unittest
from core.forecast_ai.strategy import (
    StrategyEngine, StrategyPlan, StrategyResult, StrategyCategory,
    StrategyPlanner, StrategyTemplates, TimelineGenerator, StrategyScorer,
    StrategyFormatter
)
from core.forecast_ai.recommendations import Recommendation, Category, Priority, Difficulty, RecommendationResult

class TestStrategy(unittest.TestCase):
    def setUp(self):
        self.engine = StrategyEngine()
        self.recommendations = [
            Recommendation(
                id="rec-1",
                title="Improve Quality",
                description="Increase QA",
                category=Category.QUALITY,
                priority=Priority.HIGH,
                difficulty=Difficulty.MEDIUM,
                optimization_score=0.5,
                metadata={"field": "quality"}
            ),
            Recommendation(
                id="rec-2",
                title="Increase Competency",
                description="Training",
                category=Category.COMPETENCY,
                priority=Priority.HIGH,
                difficulty=Difficulty.EASY,
                optimization_score=0.3,
                metadata={"field": "competency"}
            ),
            Recommendation(
                id="rec-3",
                title="Improve Attendance",
                description="Staffing",
                category=Category.ATTENDANCE,
                priority=Priority.MEDIUM,
                difficulty=Difficulty.HARD,
                optimization_score=0.8,
                metadata={"field": "attendance"}
            )
        ]
        self.rec_result = RecommendationResult(
            success=True,
            recommendations=self.recommendations,
            warnings=[],
            errors=[]
        )

    def test_strategy_generation(self):
        result = self.engine.generate(self.rec_result)
        self.assertTrue(result.success)
        self.assertGreater(len(result.strategies), 0)
        self.assertIsNotNone(result.best_strategy)
        # Check that strategies have required fields
        for s in result.strategies:
            self.assertIsNotNone(s.id)
            self.assertIsNotNone(s.name)
            self.assertIsNotNone(s.description)
            self.assertIsInstance(s.category, StrategyCategory)

    def test_deterministic_ids(self):
        result1 = self.engine.generate(self.rec_result)
        result2 = self.engine.generate(self.rec_result)
        self.assertEqual(
            result1.strategies[0].id,
            result2.strategies[0].id
        )

    def test_grouping(self):
        planner = StrategyPlanner()
        grouped = planner.group_recommendations(self.recommendations)
        # Should have quality, competency, attendance categories
        self.assertIn(StrategyCategory.QUALITY, grouped)
        self.assertIn(StrategyCategory.TRAINING, grouped)
        self.assertIn(StrategyCategory.STAFFING, grouped)

    def test_timeline_generation(self):
        generator = TimelineGenerator()
        milestones = generator.generate([
            (1, "Kickoff", 0.1),
            (2, "Execute", 0.5),
            (3, "Review", 0.9)
        ])
        self.assertEqual(len(milestones), 3)
        self.assertEqual(milestones[0].week, 1)
        self.assertEqual(milestones[0].expected_progress, 0.1)

    def test_scoring(self):
        scorer = StrategyScorer()
        s1 = StrategyPlan(
            id="s1", name="A", description="",
            category=StrategyCategory.GENERAL, priority="Critical",
            estimated_disruption=0.1, estimated_complexity=0.1,
            confidence=0.9, estimated_duration_weeks=2,
            objective="", recommendations=[], timeline=[], milestones=[], risks=[]
        )
        s2 = StrategyPlan(
            id="s2", name="B", description="",
            category=StrategyCategory.GENERAL, priority="Low",
            estimated_disruption=0.9, estimated_complexity=0.9,
            confidence=0.1, estimated_duration_weeks=10,
            objective="", recommendations=[], timeline=[], milestones=[], risks=[]
        )
        score1 = scorer.score(s1)
        score2 = scorer.score(s2)
        self.assertLess(score1, score2)  # s1 should be better (lower score)

    def test_formatter(self):
        strategies = [
            StrategyPlan(
                id="s1", name="Test Strategy", description="Desc",
                category=StrategyCategory.GENERAL, priority="High",
                estimated_operations_health=85.0, estimated_nps=72.0,
                estimated_duration_weeks=4, estimated_complexity=0.5,
                estimated_disruption=0.3, confidence=0.7,
                objective="", recommendations=[], timeline=[], milestones=[], risks=[]
            )
        ]
        text = StrategyFormatter.to_text(strategies)
        self.assertIn("Test Strategy", text)
        markdown = StrategyFormatter.to_markdown(strategies)
        self.assertIn("## Test Strategy", markdown)
        as_dict = StrategyFormatter.to_dict(strategies)
        self.assertEqual(as_dict[0]["name"], "Test Strategy")
        json_output = StrategyFormatter.to_json(strategies)
        self.assertIn("Test Strategy", json_output)

if __name__ == '__main__':
    unittest.main()
