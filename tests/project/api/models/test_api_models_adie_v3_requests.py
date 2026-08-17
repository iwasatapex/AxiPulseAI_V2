"""ADIE V3 request sample-bounds validation (Phase 13)."""

import pytest

from api.models.adie_v3_requests import (
    ADIEV3DecisionRequest,
    DEFAULT_SAMPLES,
    MIN_SAMPLES,
    MAX_SAMPLES,
)


def _base(**overrides):
    payload = {"observations": [0.1, 0.2, 0.3], "baseline": 0.8}
    payload.update(overrides)
    return payload


def test_default_samples_is_10000():
    req = ADIEV3DecisionRequest(**_base())
    assert req.samples == DEFAULT_SAMPLES == 10_000


def test_minimum_samples_accepted():
    req = ADIEV3DecisionRequest(**_base(samples=MIN_SAMPLES))
    assert req.samples == MIN_SAMPLES


def test_maximum_samples_accepted():
    req = ADIEV3DecisionRequest(**_base(samples=MAX_SAMPLES))
    assert req.samples == MAX_SAMPLES


def test_above_maximum_rejected():
    with pytest.raises(Exception):
        ADIEV3DecisionRequest(**_base(samples=MAX_SAMPLES + 1))


def test_below_minimum_rejected():
    with pytest.raises(Exception):
        ADIEV3DecisionRequest(**_base(samples=0))
