import unittest

from core.forecast_ai.models import ForecastRequest

from core.forecast_ai.engines.confidence_engine import ConfidenceEngine
from core.forecast_ai.engines.explainability_engine import ExplainabilityEngine
from core.forecast_ai.engines.risk_engine import RiskEngine


from dataclasses import dataclass, field

@dataclass
class DummyResult:
    success: bool = True
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    value: str = "ok"


class DummyConfidenceCore:
    def evaluate(self, **kwargs):
        return DummyResult()


class DummyExplainCore:
    def explain(self, **kwargs):
        return DummyResult()


class DummyRiskCore:
    def evaluate(self, **kwargs):
        return DummyResult()


class TestEngineWrappers(unittest.TestCase):

    def test_confidence_missing_parameters(self):
        engine = ConfidenceEngine(core=DummyConfidenceCore())
        req = ForecastRequest(operation="confidence")
        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing parameters", resp.errors[0])

    def test_confidence_success(self):
        engine = ConfidenceEngine(core=DummyConfidenceCore())
        req = ForecastRequest(
            operation="confidence",
            parameters={}
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "ConfidenceEngine")
        self.assertIsNotNone(resp.payload)

    def test_explainability_missing_parameters(self):
        engine = ExplainabilityEngine(core=DummyExplainCore())
        req = ForecastRequest(operation="explain")

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing parameters", resp.errors[0])

    def test_explainability_success(self):
        engine = ExplainabilityEngine(core=DummyExplainCore())
        req = ForecastRequest(
            operation="explain",
            parameters={}
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "ExplainabilityEngine")

    def test_risk_missing_parameters(self):
        engine = RiskEngine(core=DummyRiskCore())
        req = ForecastRequest(operation="risk")

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing parameters", resp.errors[0])

    def test_risk_success(self):
        engine = RiskEngine(core=DummyRiskCore())
        req = ForecastRequest(
            operation="risk",
            parameters={}
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "RiskEngine")


if __name__ == "__main__":
    unittest.main()


from core.forecast_ai.engines.recommendation_engine import RecommendationEngine
from core.forecast_ai.engines.reverse_optimizer import ReverseOptimizer
from core.forecast_ai.engines.strategy_engine import StrategyEngine


class DummyOptimizer:
    def optimize(self, request):
        from core.forecast_ai.optimization import OptimizationResult
        return OptimizationResult(
            success=True,
            best_solution=None,
            iterations=1,
            warnings=[],
            errors=[],
            metadata={}
        )


class DummyRecCore:
    def generate(self, result):
        from core.forecast_ai.recommendations import RecommendationResult
        return RecommendationResult(
            success=True,
            recommendations=[],
            warnings=[],
            errors=[],
            metadata={}
        )


class DummyStrategyCore:
    def generate(self, result):
        from core.forecast_ai.strategy import StrategyResult
        return StrategyResult(
            success=True,
            strategies=[],
            warnings=[],
            errors=[],
            metadata={}
        )


class TestAdditionalEngineWrappers(unittest.TestCase):

    def test_recommendation_missing_parameters(self):
        engine = RecommendationEngine()
        req = ForecastRequest(operation="recommend")

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing parameters", resp.errors[0])


    def test_reverse_optimizer_missing_parameters(self):
        engine = ReverseOptimizer()
        req = ForecastRequest(operation="reverse_optimize")

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing parameters", resp.errors[0])


    def test_strategy_missing_parameters(self):
        engine = StrategyEngine()
        req = ForecastRequest(operation="strategy")

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing parameters", resp.errors[0])


from core.forecast_ai.engines.reverse_optimizer import ReverseOptimizer
from core.forecast_ai.engines.recommendation_engine import RecommendationEngine
from core.forecast_ai.engines.strategy_engine import StrategyEngine


from dataclasses import dataclass, field


@dataclass
class DummyOptResult:
    success: bool = True
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    best_solution: object = None


class DummyOptimizer:
    def optimize(self, request):
        return DummyOptResult()


class DummyRecommendation:
    def generate(self, result):
        class Result:
            success = True
            recommendations = []
            warnings = []
            errors = []
            metadata = {}
        return Result()


class DummyStrategy:
    def generate(self, result):
        class Result:
            success = True
            strategies = []
            best_strategy = None
            warnings = []
            errors = []
            metadata = {}
        return Result()


class TestOptimizationWrapperEngines(unittest.TestCase):

    def test_reverse_optimizer_missing_target(self):
        engine = ReverseOptimizer(optimizer_core=DummyOptimizer())

        req = ForecastRequest(
            operation="reverse_optimize",
            parameters={
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing 'target_oh' or 'target_nps'", resp.errors[0])


    def test_reverse_optimizer_success(self):
        engine = ReverseOptimizer(
            optimizer_core=DummyOptimizer()
        )

        req = ForecastRequest(
            operation="reverse_optimize",
            parameters={
                "target_oh": 90,
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "ReverseOptimizer")


    def test_recommendation_success(self):
        engine = RecommendationEngine(
            optimizer_core=DummyOptimizer(),
            rec_engine=DummyRecommendation()
        )

        req = ForecastRequest(
            operation="recommend",
            parameters={
                "target_oh": 90,
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "RecommendationEngine")


    def test_strategy_missing_recommendation(self):
        engine = StrategyEngine(
            strategy_core=DummyStrategy()
        )

        req = ForecastRequest(
            operation="strategy",
            parameters={}
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing 'recommendation_result'", resp.errors[0])


    def test_strategy_success_dict_input(self):
        engine = StrategyEngine(
            strategy_core=DummyStrategy()
        )

        req = ForecastRequest(
            operation="strategy",
            parameters={
                "recommendation_result": {
                    "success": True,
                    "recommendations": [],
                    "warnings": [],
                    "errors": [],
                    "metadata": {}
                }
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "StrategyEngine")


from core.forecast_ai.engines.trend_engine import TrendEngine
from core.forecast_ai.trends.models import TrendResult, TrendAnalysis


class DummyTrendCore:
    def analyze(self, series):
        return TrendResult(
            success=True,
            analyses=[],
            warnings=[],
            errors=[],
            metadata={}
        )


class TestTrendEngineWrapper(unittest.TestCase):

    def test_trend_missing_parameters(self):
        engine = TrendEngine(trend_core=DummyTrendCore())

        req = ForecastRequest(
            operation="trend",
            parameters=None
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing parameters", resp.errors[0])


    def test_trend_missing_series(self):
        engine = TrendEngine(trend_core=DummyTrendCore())

        req = ForecastRequest(
            operation="trend",
            parameters={}
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing 'series'", resp.errors[0])


    def test_trend_invalid_series_item(self):
        engine = TrendEngine(trend_core=DummyTrendCore())

        req = ForecastRequest(
            operation="trend",
            parameters={
                "series": ["bad"]
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)


    def test_trend_missing_metric(self):
        engine = TrendEngine(trend_core=DummyTrendCore())

        req = ForecastRequest(
            operation="trend",
            parameters={
                "series": [
                    {
                        "values": [1,2],
                        "timestamps": ["d1","d2"]
                    }
                ]
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)


    def test_trend_success(self):
        engine = TrendEngine(trend_core=DummyTrendCore())

        req = ForecastRequest(
            operation="trend",
            parameters={
                "series": [
                    {
                        "metric": "quality",
                        "values": [80,81,82],
                        "timestamps": ["d1","d2","d3"]
                    }
                ]
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "TrendEngine")


from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
from core.forecast_ai.engines.sensitivity_engine import SensitivityEngine
from core.forecast_ai.engines.trend_engine import TrendEngine
from core.forecast_ai.engines.risk_engine import RiskEngine
from core.forecast_ai.models import PredictionResult


class DummyPredictionFailureService:
    def predict(self, request):
        return PredictionResult(
            operations_health=None,
            nps=None,
            warnings=[],
            errors=["prediction failed"]
        )


class DummyScenarioManager:
    def apply_scenarios(self, state, day=1):
        return state


class DummyExplodingScenario:
    def apply_scenarios(self, state, day=1):
        raise RuntimeError("scenario failure")


class TestForecastOrchestratorBranches(unittest.TestCase):

    def test_forecast_negative_horizon(self):
        engine = ForecastOrchestrator()

        req = ForecastRequest(
            operation="forecast",
            horizon=-1,
            parameters={
                "state": {
                    "quality": 85,
                    "competency": 78,
                    "release": 92,
                    "transfer": 12,
                    "attendance": 90
                }
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Horizon must be at least 1", resp.errors[0])


    def test_forecast_prediction_error_path(self):
        engine = ForecastOrchestrator(
            prediction_service=DummyPredictionFailureService(),
            scenario_manager=DummyScenarioManager()
        )

        req = ForecastRequest(
            operation="forecast",
            horizon=1,
            parameters={
                "state": {
                    "quality": 85,
                    "competency": 78,
                    "release": 92,
                    "transfer": 12,
                    "attendance": 90
                }
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)
        self.assertEqual(len(resp.payload["timeline"]), 1)
        self.assertIsNone(resp.payload["timeline"][0]["operations_health"])
        self.assertIn("prediction failed", resp.errors)


    def test_forecast_exception_path(self):
        engine = ForecastOrchestrator(
            scenario_manager=DummyExplodingScenario()
        )

        req = ForecastRequest(
            operation="forecast",
            horizon=1,
            parameters={
                "state": {
                    "quality": 85,
                    "competency": 78,
                    "release": 92,
                    "transfer": 12,
                    "attendance": 90
                }
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)
        self.assertEqual(len(resp.payload["timeline"]), 1)
        self.assertIn("scenario failure", resp.errors[0])



class ExplodingOptimizer:
    def optimize(self, request):
        raise RuntimeError("optimizer failed")


class ExplodingStrategy:
    def generate(self, result):
        raise RuntimeError("strategy failed")


class TestRemainingEngineBranches(unittest.TestCase):

    def test_recommendation_missing_state(self):
        engine = RecommendationEngine(
            optimizer_core=DummyOptimizer(),
            rec_engine=DummyRecommendation()
        )

        req = ForecastRequest(
            operation="recommend",
            parameters={
                "target_oh": 90
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing 'state'", resp.errors[0])


    def test_recommendation_optimizer_exception(self):
        engine = RecommendationEngine(
            optimizer_core=ExplodingOptimizer(),
            rec_engine=DummyRecommendation()
        )

        req = ForecastRequest(
            operation="recommend",
            parameters={
                "target_oh": 90,
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Optimization error", resp.errors[0])


    def test_reverse_optimizer_missing_state(self):
        engine = ReverseOptimizer(
            optimizer_core=DummyOptimizer()
        )

        req = ForecastRequest(
            operation="reverse_optimize",
            parameters={
                "target_oh": 90
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing 'state'", resp.errors[0])


    def test_reverse_optimizer_exception(self):
        engine = ReverseOptimizer(
            optimizer_core=ExplodingOptimizer()
        )

        req = ForecastRequest(
            operation="reverse_optimize",
            parameters={
                "target_oh": 90,
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Optimization error", resp.errors[0])


    def test_strategy_exception(self):
        engine = StrategyEngine(
            strategy_core=ExplodingStrategy()
        )

        req = ForecastRequest(
            operation="strategy",
            parameters={
                "recommendation_result": {
                    "success": True,
                    "recommendations": [],
                    "warnings": [],
                    "errors": [],
                    "metadata": {}
                }
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Strategy error", resp.errors[0])


    def test_strategy_object_recommendation_path(self):
        engine = StrategyEngine(
            strategy_core=DummyStrategy()
        )

        from core.forecast_ai.recommendations import RecommendationResult

        rec_object = RecommendationResult(
            success=True,
            recommendations=[],
            warnings=[],
            errors=[],
            metadata={}
        )

        req = ForecastRequest(
            operation="strategy",
            parameters={
                "recommendation_result": rec_object
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)



class ExplodingOptimizer:
    def optimize(self, request):
        raise RuntimeError("optimizer failed")


class ExplodingStrategy:
    def generate(self, result):
        raise RuntimeError("strategy failed")


class TestRemainingEngineBranches(unittest.TestCase):

    def test_recommendation_missing_state(self):
        engine = RecommendationEngine(
            optimizer_core=DummyOptimizer(),
            rec_engine=DummyRecommendation()
        )

        req = ForecastRequest(
            operation="recommend",
            parameters={
                "target_oh": 90
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing 'state'", resp.errors[0])


    def test_recommendation_optimizer_exception(self):
        engine = RecommendationEngine(
            optimizer_core=ExplodingOptimizer(),
            rec_engine=DummyRecommendation()
        )

        req = ForecastRequest(
            operation="recommend",
            parameters={
                "target_oh": 90,
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Optimization error", resp.errors[0])


    def test_reverse_optimizer_missing_state(self):
        engine = ReverseOptimizer(
            optimizer_core=DummyOptimizer()
        )

        req = ForecastRequest(
            operation="reverse_optimize",
            parameters={
                "target_oh": 90
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing 'state'", resp.errors[0])


    def test_reverse_optimizer_exception(self):
        engine = ReverseOptimizer(
            optimizer_core=ExplodingOptimizer()
        )

        req = ForecastRequest(
            operation="reverse_optimize",
            parameters={
                "target_oh": 90,
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Optimization error", resp.errors[0])


    def test_strategy_exception(self):
        engine = StrategyEngine(
            strategy_core=ExplodingStrategy()
        )

        req = ForecastRequest(
            operation="strategy",
            parameters={
                "recommendation_result": {
                    "success": True,
                    "recommendations": [],
                    "warnings": [],
                    "errors": [],
                    "metadata": {}
                }
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Strategy error", resp.errors[0])


    def test_strategy_object_recommendation_path(self):
        engine = StrategyEngine(
            strategy_core=DummyStrategy()
        )

        from core.forecast_ai.recommendations import RecommendationResult

        rec_object = RecommendationResult(
            success=True,
            recommendations=[],
            warnings=[],
            errors=[],
            metadata={}
        )

        req = ForecastRequest(
            operation="strategy",
            parameters={
                "recommendation_result": rec_object
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)



class ExplodingOptimizer:
    def optimize(self, request):
        raise RuntimeError("optimizer failed")


class ExplodingStrategy:
    def generate(self, result):
        raise RuntimeError("strategy failed")


class TestRemainingEngineBranches(unittest.TestCase):

    def test_recommendation_missing_state(self):
        engine = RecommendationEngine(
            optimizer_core=DummyOptimizer(),
            rec_engine=DummyRecommendation()
        )

        req = ForecastRequest(
            operation="recommend",
            parameters={
                "target_oh": 90
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing 'state'", resp.errors[0])


    def test_recommendation_optimizer_exception(self):
        engine = RecommendationEngine(
            optimizer_core=ExplodingOptimizer(),
            rec_engine=DummyRecommendation()
        )

        req = ForecastRequest(
            operation="recommend",
            parameters={
                "target_oh": 90,
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Optimization error", resp.errors[0])


    def test_reverse_optimizer_missing_state(self):
        engine = ReverseOptimizer(
            optimizer_core=DummyOptimizer()
        )

        req = ForecastRequest(
            operation="reverse_optimize",
            parameters={
                "target_oh": 90
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing 'state'", resp.errors[0])


    def test_reverse_optimizer_exception(self):
        engine = ReverseOptimizer(
            optimizer_core=ExplodingOptimizer()
        )

        req = ForecastRequest(
            operation="reverse_optimize",
            parameters={
                "target_oh": 90,
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Optimization error", resp.errors[0])


    def test_strategy_exception(self):
        engine = StrategyEngine(
            strategy_core=ExplodingStrategy()
        )

        req = ForecastRequest(
            operation="strategy",
            parameters={
                "recommendation_result": {
                    "success": True,
                    "recommendations": [],
                    "warnings": [],
                    "errors": [],
                    "metadata": {}
                }
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Strategy error", resp.errors[0])


    def test_strategy_object_recommendation_path(self):
        engine = StrategyEngine(
            strategy_core=DummyStrategy()
        )

        from core.forecast_ai.recommendations import RecommendationResult

        rec_object = RecommendationResult(
            success=True,
            recommendations=[],
            warnings=[],
            errors=[],
            metadata={}
        )

        req = ForecastRequest(
            operation="strategy",
            parameters={
                "recommendation_result": rec_object
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)



class DummyTrend:
    def analyze(self, series):
        from core.forecast_ai.trends import TrendResult
        return TrendResult(
            success=True,
            analyses=[],
            warnings=[],
            errors=[],
            metadata={}
        )


class ExplodingTrendCore:
    def analyze(self, series):
        raise RuntimeError("trend failed")


class ExplodingSensitivityCore:
    def analyze(self, state, metrics=None):
        raise RuntimeError("sensitivity failed")


class ExplodingRiskCore:
    def evaluate(self, **kwargs):
        raise RuntimeError("risk failed")


class TestAnalyticsEngineBranches(unittest.TestCase):

    def test_trend_skips_empty_series(self):
        engine = TrendEngine(
            trend_core=DummyTrend()
        )

        req = ForecastRequest(
            operation="trend",
            parameters={
                "series": [
                    {
                        "metric": "NPS",
                        "values": [],
                        "timestamps": []
                    }
                ]
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)


    def test_trend_core_exception(self):
        engine = TrendEngine(
            trend_core=ExplodingTrendCore()
        )

        req = ForecastRequest(
            operation="trend",
            parameters={
                "series": [
                    {
                        "metric": "NPS",
                        "values": [1,2],
                        "timestamps": ["d1","d2"]
                    }
                ]
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Trend analysis error", resp.errors[0])


    def test_sensitivity_missing_parameters(self):
        engine = SensitivityEngine()

        req = ForecastRequest(
            operation="sensitivity",
            parameters=None
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing parameters", resp.errors[0])


    def test_sensitivity_core_exception(self):
        engine = SensitivityEngine(
            sen_core=ExplodingSensitivityCore()
        )

        req = ForecastRequest(
            operation="sensitivity",
            parameters={
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Sensitivity error", resp.errors[0])


    def test_risk_core_exception(self):
        engine = RiskEngine(
            core=ExplodingRiskCore()
        )

        req = ForecastRequest(
            operation="risk",
            parameters={}
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Risk error", resp.errors[0])



class DummyTrend:
    def analyze(self, series):
        from core.forecast_ai.trends import TrendResult
        return TrendResult(
            success=True,
            analyses=[],
            warnings=[],
            errors=[],
            metadata={}
        )


class ExplodingTrendCore:
    def analyze(self, series):
        raise RuntimeError("trend failed")


class ExplodingSensitivityCore:
    def analyze(self, state, metrics=None):
        raise RuntimeError("sensitivity failed")


class ExplodingRiskCore:
    def evaluate(self, **kwargs):
        raise RuntimeError("risk failed")


class TestAnalyticsEngineBranches(unittest.TestCase):

    def test_trend_skips_empty_series(self):
        engine = TrendEngine(
            trend_core=DummyTrend()
        )

        req = ForecastRequest(
            operation="trend",
            parameters={
                "series": [
                    {
                        "metric": "NPS",
                        "values": [],
                        "timestamps": []
                    }
                ]
            }
        )

        resp = engine.execute(req)

        self.assertTrue(resp.success)


    def test_trend_core_exception(self):
        engine = TrendEngine(
            trend_core=ExplodingTrendCore()
        )

        req = ForecastRequest(
            operation="trend",
            parameters={
                "series": [
                    {
                        "metric": "NPS",
                        "values": [1,2],
                        "timestamps": ["d1","d2"]
                    }
                ]
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Trend analysis error", resp.errors[0])


    def test_sensitivity_missing_parameters(self):
        engine = SensitivityEngine()

        req = ForecastRequest(
            operation="sensitivity",
            parameters=None
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Missing parameters", resp.errors[0])


    def test_sensitivity_core_exception(self):
        engine = SensitivityEngine(
            sen_core=ExplodingSensitivityCore()
        )

        req = ForecastRequest(
            operation="sensitivity",
            parameters={
                "state": {}
            }
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Sensitivity error", resp.errors[0])


    def test_risk_core_exception(self):
        engine = RiskEngine(
            core=ExplodingRiskCore()
        )

        req = ForecastRequest(
            operation="risk",
            parameters={}
        )

        resp = engine.execute(req)

        self.assertFalse(resp.success)
        self.assertIn("Risk error", resp.errors[0])

