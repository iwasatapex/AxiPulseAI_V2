from abc import ABC, abstractmethod
from .models import ForecastRequest, ForecastResponse

class ForecastAIEngine(ABC):
    @abstractmethod
    def execute(self, request: ForecastRequest) -> ForecastResponse:
        pass
