"""ForecastAI - Intelligent Forecasting & Decision Engine"""
__version__ = "0.1.0"

# Export prediction components
from .prediction import PredictorProvider, PredictionService

# Export key components
from .prediction import PredictorProvider, PredictionService

# Export key components
from .prediction import PredictorProvider, PredictionService
# Export state components
from .state import OperationalState, StateEvolutionEngine

# Export scenario components
from .scenarios import ScenarioManager, ScenarioRegistry, Scenario, Modifier, ModifierType

# Export optimization components
from .optimization import ReverseOptimizer, TargetGoal, Constraint, ConstraintType

# Export recommendation components
from .recommendations import RecommendationEngine, Recommendation, RecommendationResult, Category, Priority, Difficulty

# Export strategy components
from .strategy import StrategyEngine, StrategyPlan, StrategyResult, StrategyCategory, Milestone

# Export trend components
from .trends import TrendEngine, TrendSeries, TrendAnalysis, TrendResult

# Export sensitivity components
from .sensitivity import SensitivityEngine, SensitivityAnalysis, SensitivityResult

# Export confidence components
from .confidence import ConfidenceEngine, ConfidenceResult, ConfidenceAnalysis

# Export risk components
from .risk import RiskEngine, RiskResult, RiskAnalysis, RiskFactor

# Export explainability components
from .explainability import ExplainabilityEngine, ExplainabilityResult, Explanation

# Export reporting components
from .reporting import ReportEngine, ReportResult, ReportType
