"""V2.3 compliance tests: NPS is computed from actual survey counts.

Rule 14: NPS = %promoters - %detractors, where promoters are scores 9-10
and detractors are scores 0-6. NPS is bounded to -100..100.
"""
import pytest

from core.nps_predictor.bayesian_distribution import nps_from_score_counts


def test_nps_from_all_promoters_is_plus_100():
    # Scores 9 and 10 populated; nothing else.
    r = nps_from_score_counts([0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5])
    assert r["promoters"] == 10
    assert r["detractors"] == 0
    assert r["nps"] == pytest.approx(100.0)


def test_nps_from_all_detractors_is_minus_100():
    # All 11 counts on scores 0-6 (detractors).
    r = nps_from_score_counts([2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0])
    assert r["detractors"] == 11
    assert r["promoters"] == 0
    assert r["nps"] == pytest.approx(-100.0)


def test_nps_is_pct_promoters_minus_pct_detractors():
    # 2 detractors (scores 0,1), 9 promoters (scores 9,10), 11 total.
    r = nps_from_score_counts([1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 4])
    assert r["detractors"] == 2
    assert r["promoters"] == 9
    assert r["nps"] == pytest.approx((9 - 2) / 11 * 100.0)


def test_nps_stays_within_bounds():
    for counts in [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5],
        [2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 4],
        [0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0],
    ]:
        r = nps_from_score_counts(counts)
        assert -100.0 <= r["nps"] <= 100.0
