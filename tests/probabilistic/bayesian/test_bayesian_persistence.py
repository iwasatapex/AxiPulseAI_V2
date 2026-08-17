from pathlib import Path

import pytest

from core.bayesian.feedback import BayesianFeedbackState
from core.bayesian.persistence import (
    load_state,
    save_state,
    state_from_dict,
    state_to_dict,
)


def test_state_round_trip():
    state = BayesianFeedbackState(
        prior_mean=0.7,
        prior_strength=5.0,
        observations=3,
        successes=2.0,
        failures=1.0,
        metadata={"source": "test"},
    )

    payload = state_to_dict(state)
    restored = state_from_dict(payload)

    assert restored == state


def test_json_file_round_trip(tmp_path: Path):
    state = BayesianFeedbackState(
        prior_mean=0.65,
        prior_strength=4.0,
        observations=5,
        successes=4.0,
        failures=1.0,
        metadata={"version": 1},
    )

    path = tmp_path / "bayesian_state.json"

    saved = save_state(state, path)

    assert saved == path
    assert path.is_file()

    restored = load_state(path)

    assert restored == state


def test_save_creates_parent_directories(tmp_path: Path):
    state = BayesianFeedbackState()

    path = (
        tmp_path
        / "nested"
        / "state"
        / "bayesian.json"
    )

    save_state(state, path)

    assert path.is_file()
    assert load_state(path) == state


def test_missing_state_file_is_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_state(
            tmp_path / "missing.json"
        )


def test_invalid_payload_is_rejected():
    with pytest.raises(TypeError):
        state_from_dict([])


def test_unknown_fields_are_rejected():
    with pytest.raises(ValueError):
        state_from_dict(
            {
                "prior_mean": 0.5,
                "unexpected": True,
            }
        )


def test_persistence_does_not_mutate_state(tmp_path: Path):
    state = BayesianFeedbackState(
        observations=2,
        successes=1.0,
        failures=1.0,
    )

    before = state

    save_state(
        state,
        tmp_path / "state.json",
    )

    assert state == before
