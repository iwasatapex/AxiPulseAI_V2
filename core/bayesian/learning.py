from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .feedback import BayesianFeedbackState


@dataclass(frozen=True)
class BayesianObservation:
    """
    Immutable record of an observed outcome.

    prediction is informational only. It is never fed back into the
    same prediction step.
    """

    prediction_id: str
    outcome: float
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None


class BayesianLearningLoop:
    """
    Explicit prediction → outcome → update boundary.

    A prediction must exist before its corresponding outcome can
    update the Bayesian state.

    No same-step prediction mutation is performed.
    """

    def __init__(
        self,
        state: BayesianFeedbackState | None = None,
    ) -> None:
        self._state = state or BayesianFeedbackState()

    @property
    def state(self) -> BayesianFeedbackState:
        return self._state

    def record_outcome(
        self,
        observation: BayesianObservation,
    ) -> BayesianFeedbackState:
        """
        Incorporate a previously observed outcome.

        The prediction itself is not changed or recomputed here.
        """
        self._state = self._state.update(
            observation.outcome
        )

        return self._state

    def record_outcomes(
        self,
        observations: list[BayesianObservation],
    ) -> BayesianFeedbackState:
        """
        Incorporate historical outcomes in supplied order.
        """
        for observation in observations:
            self.record_outcome(observation)

        return self._state

    def posterior(self) -> dict[str, float]:
        return self._state.posterior()
