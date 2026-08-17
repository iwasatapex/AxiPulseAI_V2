from datetime import datetime

import pytest

from core.common.temporal_contract import (
    assert_forecast_boundary,
    assert_known_at_cutoff,
)


def test_target_must_be_after_cutoff():
    cutoff = datetime(2026, 1, 10)
    target = datetime(2026, 1, 11)

    assert_forecast_boundary(cutoff, target)


def test_same_day_target_is_rejected():
    cutoff = datetime(2026, 1, 10)

    with pytest.raises(ValueError):
        assert_forecast_boundary(cutoff, cutoff)


def test_past_information_is_allowed():
    cutoff = datetime(2026, 1, 10)

    assert_known_at_cutoff(
        datetime(2026, 1, 10),
        cutoff,
        field_name="operational_health",
    )


def test_future_information_is_rejected():
    cutoff = datetime(2026, 1, 10)

    with pytest.raises(ValueError, match="Temporal leakage"):
        assert_known_at_cutoff(
            datetime(2026, 1, 11),
            cutoff,
            field_name="operational_health",
        )


def test_future_oh_cannot_be_used_for_nps_cutoff():
    cutoff = datetime(2026, 1, 10)

    with pytest.raises(ValueError):
        assert_known_at_cutoff(
            datetime(2026, 1, 11),
            cutoff,
            field_name="operational_health",
        )
