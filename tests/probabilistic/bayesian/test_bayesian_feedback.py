import pytest

from core.bayesian.feedback import BayesianFeedbackState
from core.bayesian.learning import (
    BayesianLearningLoop,
    BayesianObservation,
)


def test_initial_state_uses_prior():
    state = BayesianFeedbackState(
        prior_mean=0.6,
        prior_strength=4.0,
    )

    assert state.posterior_mean == pytest.approx(0.6)
    assert state.observations == 0


def test_single_outcome_updates_posterior():
    state = BayesianFeedbackState(
        prior_mean=0.5,
        prior_strength=2.0,
    )

    updated = state.update(1.0)

    assert state.observations == 0
    assert updated.observations == 1
    assert updated.posterior_mean > state.posterior_mean


def test_failure_moves_posterior_down():
    state = BayesianFeedbackState(
        prior_mean=0.5,
        prior_strength=2.0,
    )

    updated = state.update(0.0)

    assert updated.posterior_mean < state.posterior_mean


def test_fractional_observation_is_supported():
    state = BayesianFeedbackState()

    updated = state.update(0.25)

    assert updated.observations == 1
    assert updated.successes == pytest.approx(0.25)
    assert updated.failures == pytest.approx(0.75)


def test_invalid_outcome_is_rejected():
    state = BayesianFeedbackState()

    with pytest.raises(ValueError):
        state.update(-0.1)

    with pytest.raises(ValueError):
        state.update(1.1)


def test_update_many_does_not_mutate_original():
    state = BayesianFeedbackState()

    updated = state.update_many([1.0, 1.0, 0.0])

    assert state.observations == 0
    assert updated.observations == 3


def test_learning_loop_requires_observed_outcome():
    loop = BayesianLearningLoop()

    observation = BayesianObservation(
        prediction_id="prediction-1",
        outcome=1.0,
    )

    state = loop.record_outcome(observation)

    assert state.observations == 1
    assert state.posterior_mean > 0.5


def test_learning_loop_keeps_prediction_metadata_separate():
    loop = BayesianLearningLoop()

    observation = BayesianObservation(
        prediction_id="prediction-42",
        outcome=0.0,
        metadata={"predicted_probability": 0.9},
    )

    state = loop.record_outcome(observation)

    assert state.observations == 1
    assert state.posterior_mean < 0.5


def test_posterior_summary_is_explicit():
    state = BayesianFeedbackState()

    summary = state.posterior()

    assert set(summary) == {
        "mean",
        "alpha",
        "beta",
        "strength",
    }
