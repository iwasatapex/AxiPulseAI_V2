import datetime
from typing import Optional, List, Dict, Any
from .models import ForecastRequest, ForecastResponse, OperationType
from .base_engine import ForecastAIEngine
from .engines import (
    ForecastOrchestrator,
    ReverseOptimizer,
    RecommendationEngine,
    StrategyEngine
)
# Other engines (not yet moved) could be imported here if they exist later

class Validator:
    def __init__(self):
        self.operations = [op.value for op in OperationType]
        self.supported_horizons = [1, 5, 7, 30, 90, 180, 365]
    def validate(self, request: ForecastRequest) -> List[str]:
        errors = []
        if not request.operation:
            errors.append("Missing 'operation' field")
        elif request.operation not in self.operations:
            errors.append(f"Unsupported operation: {request.operation}")
        if request.horizon is not None and request.horizon not in self.supported_horizons:
            errors.append(f"Unsupported horizon: {request.horizon}")

        return errors

class Router:
    def __init__(self, engine_registry: Optional[Dict[str, ForecastAIEngine]] = None):
        if engine_registry:
            self.engine_map = engine_registry
        else:
            self.engine_map = {
                OperationType.FORECAST.value: ForecastOrchestrator(),
                OperationType.REVERSE_OPTIMIZE.value: ReverseOptimizer(),
                OperationType.RECOMMEND.value: RecommendationEngine(),
                OperationType.STRATEGY.value: StrategyEngine(),
                # TODO: add other engines when they are implemented and moved to engines/
                # OperationType.SCENARIO.value: ScenarioManager(),
                # etc.
            }
    def route(self, operation: str) -> Optional[ForecastAIEngine]:
        return self.engine_map.get(operation)

class Dispatcher:
    def dispatch(self, engine: ForecastAIEngine, request: ForecastRequest) -> ForecastResponse:
        try:
            return engine.execute(request)
        except Exception as e:
            return ForecastResponse(
                success=False, operation="unknown", engine="Dispatcher",
                timestamp=datetime.datetime.now().isoformat(),
                warnings=[], errors=[str(e)], metadata={}, payload=None
            )

class ResponseBuilder:
    @staticmethod
    def build(response: ForecastResponse) -> ForecastResponse:
        required = ['success','operation','engine','timestamp','warnings','errors','metadata']
        for field in required:
            if not hasattr(response, field):
                setattr(response, field, [] if field in ['warnings','errors'] else None)
        return response

class ForecastAIPlanner:
    def __init__(self, router: Optional[Router] = None, validator: Optional[Validator] = None):
        self.validator = validator or Validator()
        self.router = router or Router()
        self.dispatcher = Dispatcher()
        self.builder = ResponseBuilder()

    def execute(self, request: ForecastRequest) -> ForecastResponse:
        errors = self.validator.validate(request)
        if errors:
            return self._error_response(errors)
        engine = self.router.route(request.operation)
        if engine is None:
            return self._error_response([f"No engine found for operation: {request.operation}"])
        response = self.dispatcher.dispatch(engine, request)
        return self.builder.build(response)

    def validate(self, request: ForecastRequest) -> List[str]:
        return self.validator.validate(request)

    def route(self, request: ForecastRequest) -> Optional[ForecastAIEngine]:
        return self.router.route(request.operation)

    def _error_response(self, errors: List[str]) -> ForecastResponse:
        return ForecastResponse(
            success=False, operation="unknown", engine="Planner",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[], errors=errors, metadata={}, payload=None
        )
