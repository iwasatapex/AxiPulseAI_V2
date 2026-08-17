"""
StateEvolutionEngine – Pure state transition logic.
No business rules – only propagates predicted values.
"""
from typing import Optional
from .models import OperationalState
from ..models import PredictionResult

class StateEvolutionEngine:
    """
    Evolves the operational state based on a prediction result.
    Immutable: returns a new OperationalState, never mutates input.
    """
    def evolve(self, current_state: OperationalState, prediction_result: PredictionResult) -> OperationalState:
        """
        Create a new state by copying current_state and updating
        operations_health and nps if they are not None.
        """
        new_state = OperationalState(
            quality=prediction_result.quality
                if prediction_result.quality is not None
                else current_state.quality,

            competency=prediction_result.competency
                if prediction_result.competency is not None
                else current_state.competency,

            transfer=prediction_result.transfer
                if prediction_result.transfer is not None
                else current_state.transfer,

            release=prediction_result.release
                if prediction_result.release is not None
                else current_state.release,

            attendance=prediction_result.attendance
                if prediction_result.attendance is not None
                else current_state.attendance,

            operations_health=prediction_result.operations_health
                if prediction_result.operations_health is not None
                else current_state.operations_health,

            nps=prediction_result.nps
                if prediction_result.nps is not None
                else current_state.nps,

            metadata=current_state.metadata
        )
        return new_state
