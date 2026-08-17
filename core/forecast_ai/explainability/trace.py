"""
Trace builder – builds execution chain with dependencies.
"""
from typing import List, Dict, Any
from .models import ExplanationTrace

class TraceBuilder:
    @staticmethod
    def build_trace(active_components: List[str]) -> List[ExplanationTrace]:
        """
        Build execution trace in order: Prediction → Forecast → Scenario → Optimization → Recommendation → Strategy → Trend → Sensitivity → Confidence → Risk → Explainability.
        Only include components that are present.
        """
        full_chain = [
            'PredictionService',
            'ForecastOrchestrator',
            'ScenarioManager',
            'ReverseOptimizer',
            'RecommendationEngine',
            'StrategyEngine',
            'TrendEngine',
            'SensitivityEngine',
            'ConfidenceEngine',
            'RiskEngine',
            'ExplainabilityEngine'
        ]

        # Map component names to their display names and descriptions
        component_info = {
            'PredictionService': {'purpose': 'Predict OH and NPS from current state.', 'dependencies': ['OperationalState']},
            'ForecastOrchestrator': {'purpose': 'Perform recursive forecasting over horizon.', 'dependencies': ['PredictionService']},
            'ScenarioManager': {'purpose': 'Apply scenarios to modify operational state.', 'dependencies': ['ForecastOrchestrator']},
            'ReverseOptimizer': {'purpose': 'Search for operational changes to meet target.', 'dependencies': ['PredictionService', 'ScenarioManager']},
            'RecommendationEngine': {'purpose': 'Convert optimization results into recommendations.', 'dependencies': ['ReverseOptimizer']},
            'StrategyEngine': {'purpose': 'Group recommendations into operational strategies.', 'dependencies': ['RecommendationEngine']},
            'TrendEngine': {'purpose': 'Analyze KPI timelines for patterns and trends.', 'dependencies': ['ForecastOrchestrator']},
            'SensitivityEngine': {'purpose': 'Measure KPI influence on outputs.', 'dependencies': ['PredictionService']},
            'ConfidenceEngine': {'purpose': 'Evaluate confidence in outputs.', 'dependencies': ['ForecastOrchestrator', 'TrendEngine', 'SensitivityEngine', 'RecommendationEngine', 'StrategyEngine']},
            'RiskEngine': {'purpose': 'Identify operational and execution risks.', 'dependencies': ['ConfidenceEngine', 'ForecastOrchestrator', 'TrendEngine', 'SensitivityEngine', 'RecommendationEngine', 'StrategyEngine']},
            'ExplainabilityEngine': {'purpose': 'Generate explanations and traces for all outputs.', 'dependencies': ['RiskEngine', 'ConfidenceEngine', 'TrendEngine', 'SensitivityEngine', 'RecommendationEngine', 'StrategyEngine', 'ForecastOrchestrator']}
        }

        # Build the trace for components present
        traces = []
        step = 1
        for engine_name in full_chain:
            # Map component names to active_components
            # active_components are lower-case names like 'forecast', 'trend'
            # We need to map them to engine names: 'forecast' -> 'ForecastOrchestrator'
            # We'll create a mapping from component to engine name
            component_to_engine = {
                'forecast': 'ForecastOrchestrator',
                'trend': 'TrendEngine',
                'sensitivity': 'SensitivityEngine',
                'recommendation': 'RecommendationEngine',
                'strategy': 'StrategyEngine',
                'confidence': 'ConfidenceEngine',
                'risk': 'RiskEngine'
            }
            # Check if this engine is active
            is_active = False
            for comp in active_components:
                if component_to_engine.get(comp) == engine_name:
                    is_active = True
                    break
            # Always include if engine_name is 'PredictionService' or 'ExplainabilityEngine'
            if engine_name in ['PredictionService', 'ExplainabilityEngine']:
                is_active = True
            if is_active:
                info = component_info.get(engine_name, {'purpose': '', 'dependencies': []})
                traces.append(ExplanationTrace(
                    step=step,
                    engine=engine_name,
                    description=f"Executed {engine_name}.",
                    purpose=info.get('purpose', ''),
                    input_reference="previous step output" if step > 1 else "operational state",
                    output_reference=f"{engine_name} result",
                    dependencies=info.get('dependencies', []),
                    metadata={"active": True}
                ))
                step += 1
            else:
                # Skip this engine but keep step? We'll keep step increment only for active ones.
                pass
        return traces

build_trace = TraceBuilder.build_trace
