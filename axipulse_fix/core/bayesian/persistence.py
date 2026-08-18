from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .feedback import BayesianFeedbackState


def state_to_dict(
    state: BayesianFeedbackState,
) -> dict[str, Any]:
    """
    Convert Bayesian state into a JSON-safe dictionary.
    """
    return asdict(state)


def state_from_dict(
    payload: dict[str, Any],
) -> BayesianFeedbackState:
    """
    Reconstruct Bayesian state from a persisted dictionary.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")

    allowed = {
        "prior_mean",
        "prior_strength",
        "observations",
        "successes",
        "failures",
        "metadata",
    }

    unknown = set(payload) - allowed

    if unknown:
        raise ValueError(
            f"unknown Bayesian state fields: {sorted(unknown)}"
        )

    metadata = payload.get("metadata")

    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary or None")

    return BayesianFeedbackState(
        prior_mean=float(payload.get("prior_mean", 0.5)),
        prior_strength=float(payload.get("prior_strength", 2.0)),
        observations=int(payload.get("observations", 0)),
        successes=float(payload.get("successes", 0.0)),
        failures=float(payload.get("failures", 0.0)),
        metadata=metadata,
    )


def save_state(
    state: BayesianFeedbackState,
    path: str | Path,
) -> Path:
    """
    Atomically persist Bayesian state as JSON.
    """
    destination = Path(path)

    if destination.parent:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    payload = state_to_dict(state)

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary.replace(destination)

    return destination


def load_state(
    path: str | Path,
) -> BayesianFeedbackState:
    """
    Load Bayesian state from JSON.
    """
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(source)

    payload = json.loads(
        source.read_text()
    )

    return state_from_dict(payload)
