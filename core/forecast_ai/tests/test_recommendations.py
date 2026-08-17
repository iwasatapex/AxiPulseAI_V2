import unittest
from core.forecast_ai.recommendations import (
    RecommendationEngine, Recommendation, Category, Priority, Difficulty,
    RecommendationTemplates, RecommendationRanker, RecommendationFormatter
)
from core.forecast_ai.optimization import OptimizationSolution, OptimizationResult
from core.forecast_ai.prediction.provider import PredictorProvider

class TestRecommendations(unittest.TestCase):
    def setUp(self):
        self.engine = RecommendationEngine()
        # Create a dummy optimization result
        self.solution = OptimizationSolution(
            predicted_operations_health=88.0,
            predicted_nps=72.0,
            state_changes={"quality": 5.0, "competency": 3.0},
            applied_scenarios=[],
            optimization_score=0.5,
            distance_to_target=1.0,
            iterations_used=10,
            constraints_satisfied=True,
            state={"quality": 85, "competency": 73}
        )
        self.result = OptimizationResult(
            success=True,
            solutions=[self.solution],
            best_solution=self.solution,
            warnings=[],
            errors=[],
            metadata={}
        )

    def test_generate_recommendations(self):
        rec_result = self.engine.generate(self.result)
        self.assertTrue(rec_result.success)
        self.assertGreaterEqual(len(rec_result.recommendations), 1)
        # Check that each recommendation has required fields
        for rec in rec_result.recommendations:
            self.assertIsNotNone(rec.id)
            self.assertIsNotNone(rec.title)
            self.assertIsNotNone(rec.description)
            self.assertIsInstance(rec.category, Category)
            self.assertIsInstance(rec.priority, Priority)
            self.assertIsInstance(rec.difficulty, Difficulty)

    def test_ranking(self):
        recs = [
            Recommendation(id="1", title="A", description="", category=Category.QUALITY,
                           priority=Priority.HIGH, difficulty=Difficulty.EASY,
                           estimated_operations_health_gain=2.0, optimization_score=0.5),
            Recommendation(id="2", title="B", description="", category=Category.COMPETENCY,
                           priority=Priority.CRITICAL, difficulty=Difficulty.MEDIUM,
                           estimated_operations_health_gain=1.0, optimization_score=0.3),
        ]
        ranked = RecommendationRanker.rank(recs)
        self.assertEqual(ranked[0].id, "2")  # Critical priority first

    def test_formatter(self):
        recs = [
            Recommendation(id="1", title="Test", description="Desc", category=Category.QUALITY,
                           priority=Priority.HIGH, difficulty=Difficulty.EASY,
                           estimated_operations_health_gain=5.0)
        ]
        text = RecommendationFormatter.to_text(recs)
        self.assertIn("Test", text)
        markdown = RecommendationFormatter.to_markdown(recs)
        self.assertIn("## Test", markdown)
        as_dict = RecommendationFormatter.to_dict(recs)
        self.assertEqual(as_dict[0]["title"], "Test")

    def test_template_mapping(self):
        templates = RecommendationTemplates()
        title, desc, cat = templates.get_template("quality")
        self.assertEqual(cat, Category.QUALITY)
        actions = templates.get_actions("quality")
        self.assertGreater(len(actions), 0)

if __name__ == '__main__':
    unittest.main()
