"""Recommendation engine – converts optimization results into operational advice."""
from .models import Recommendation, RecommendationResult, Category, Priority, Difficulty
from .engine import RecommendationEngine
from .ranking import RecommendationRanker
from .templates import RecommendationTemplates
from .formatter import RecommendationFormatter
from .conflicts import ConflictDetector
