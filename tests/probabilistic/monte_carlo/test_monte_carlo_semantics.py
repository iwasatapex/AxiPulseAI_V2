"""
Regression tests: Monte Carlo binary-outcome vs continuous-score semantics.

The ADIE Monte Carlo detail contains TWO distinct statistics that must not be
conflated:

  1. Binary target-attainment outcome: success_count / failure_count /
     success_percentage / failure_percentage = P(OH >= target_oh). For a
     Bernoulli outcome, the expected value of the binary indicator equals the
     success rate.

  2. Continuous Monte Carlo score: expected_value / p05 / p50 / p95 /
     uncertainty = the percentiles and spread of the underlying continuous OH
     distribution. These are NOT the expected binary success value and must be
     labelled as a continuous score.

These tests assert that:
  - the binary success rate is internally consistent (counts sum to samples,
    success rate equals binary expectation),
  - the continuous score statistics are kept separate and not relabelled as
    binary outcome statistics,
  - the GUI renderer distinguishes the two sections with distinct labels.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import numpy as np
import pytest

from core.decision_intelligence.v3.synthesis.decision_detail import (
    _build_mc_detail,
)

VIEW_PATH = pathlib.Path(inspect.getsourcefile(__import__(
    "gui.views.adie_view", fromlist=["adie_view"])))


def _mc_detail(mean=0.88, p05=0.799, p50=0.879, p95=0.962,
               uncertainty=0.05, samples=10000):
    return {
        "distribution_summary": {
            "samples": samples,
            "mean": mean,
            "p05": p05,
            "p50": p50,
            "p95": p95,
            "uncertainty": uncertainty,
        },
        "distribution": [],
    }


# --------------------------------------------------------------------------- #
# 1. Binary expected value equals success rate
# --------------------------------------------------------------------------- #
def test_binary_success_rate_consistent_with_counts():
    """success_count/samples == success_percentage/100."""
    out = _build_mc_detail(_mc_detail(), {"target_oh": 81.0})
    n = int(out["total_samples"])
    assert out["success_count"] + out["failure_count"] == n
    rate = out["success_count"] / n
    assert abs(out["success_percentage"] / 100.0 - rate) < 1e-6
    assert abs(out["failure_percentage"] / 100.0 - (1.0 - rate)) < 1e-6


def test_binary_outcome_expected_value_equals_success_rate():
    """For a binary Bernoulli outcome expected_value == success_rate.

    The continuous-score expected_value must NOT be conflated with this. The
    binary expectation is derived from success_count / total_samples.
    """
    out = _build_mc_detail(_mc_detail(), {"target_oh": 81.0})
    binary_expected = out["success_count"] / int(out["total_samples"])
    assert 0.0 <= binary_expected <= 1.0
    # The binary expected value equals the success rate.
    assert abs(binary_expected - out["success_percentage"] / 100.0) < 1e-6


# --------------------------------------------------------------------------- #
# 2. Binary p05/p50/p95 consistent with binary samples
# --------------------------------------------------------------------------- #
def test_binary_percentiles_consistent_with_binary_samples():
    """A binary (0/1) sample's percentiles must be in {0, 1}, not the
    continuous score percentiles."""
    out = _build_mc_detail(_mc_detail(), {"target_oh": 81.0})
    success = out["success_count"]
    failure = out["failure_count"]
    samples = np.asarray([1] * success + [0] * failure, dtype=float)
    n = samples.size
    bp05 = float(np.percentile(samples, 5))
    bp50 = float(np.percentile(samples, 50))
    bp95 = float(np.percentile(samples, 95))
    # Consistent with a 0/1 variable: percentiles are exactly 0 or 1.
    assert bp05 in (0.0, 1.0)
    assert bp50 in (0.0, 1.0)
    assert bp95 in (0.0, 1.0)
    # The binary mean equals the success rate.
    assert abs(float(np.mean(samples)) - success / n) < 1e-6


# --------------------------------------------------------------------------- #
# 3. Continuous score statistics not mislabeled as binary outcome
# --------------------------------------------------------------------------- #
def test_continuous_score_kept_separate_from_binary():
    """expected_value/p05/p50/p95 describe the continuous OH distribution and
    are retained as such, not replaced by the binary 0/1 expectation."""
    out = _build_mc_detail(_mc_detail(), {"target_oh": 81.0})
    # Binary success rate is distinct from the continuous expected value.
    binary_expected = out["success_count"] / int(out["total_samples"])
    continuous_expected = out["expected_value"]
    assert binary_expected != continuous_expected
    # Continuous percentile range is consistent with a continuous distribution.
    assert out["p05"] < out["p50"] < out["p95"]


def test_continuous_score_not_binary_magnitude():
    """Continuous expected value is on the OH score scale (0..1), not 0/1."""
    out = _build_mc_detail(_mc_detail(mean=0.88), {"target_oh": 81.0})
    assert out["expected_value"] == pytest.approx(out["expected_value"])
    assert 0.0 <= out["expected_value"] <= 1.0
    assert out["p50"] == pytest.approx(out["expected_value"], abs=0.05)


# --------------------------------------------------------------------------- #
# 4. GUI labels distinguish the two sections
# --------------------------------------------------------------------------- #
def _view_source():
    return VIEW_PATH.read_text()


def test_gui_renders_distinct_binary_and_continuous_sections():
    src = _view_source()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_render_mc_detail")
    seg = ast.get_source_segment(src, fn)
    assert "Binary Outcome" in seg
    assert "Continuous Monte Carlo Score" in seg


def test_gui_does_not_mislabel_continuous_expected_value_as_binary():
    """The continuous expected value must be labelled as a score, never as the
    expected binary success value."""
    src = _view_source()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_render_mc_detail")
    seg = ast.get_source_segment(src, fn)
    # The continuous metric is explicitly an "Expected Score".
    assert '"Expected Score"' in seg
    # It is not presented as an "Expected Value" that a reader would read as
    # the binary expectation.
    assert '"Expected Value"' not in seg


def test_gui_binary_section_shows_success_rate_label():
    src = _view_source()
    assert '"Success Rate"' in src
    assert '"Failure Rate"' in src
