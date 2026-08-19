"""Focused regression tests for the reverse-optimizer cleanup.

After removing the obsolete ``gui.services.reverse_optimize(metric, target,
family)`` wrapper (which delegated to TargetStateEngine and competed with the
canonical OH/NPS reverse path), this file proves:

  * the GUI OH/NPS reverse view uses ONLY ``reverse_optimize_canonical()``;
  * the legacy ``gui.services.reverse_optimize`` wrapper is gone;
  * ``gui.services.find_target_state`` / TargetStateEngine remain available for
    the separate multi-KPI Target State feature;
  * ``reverse_optimize_canonical`` does not depend on TargetStateEngine /
    find_target_state;
  * canonical reverse output shape guarantees (OH + NPS, hard-bound candidates,
    canonical MC NPS interval) are preserved by the existing focused suites.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import services as svc  # noqa: E402
from gui.views import reverse_view  # noqa: E402


def test_gui_reverse_view_uses_only_reverse_optimize_canonical():
    src = inspect.getsource(reverse_view)
    assert "reverse_optimize_canonical" in src
    # It must NOT call the legacy wrapper, find_target_state, or TargetStateEngine.
    assert "svc.reverse_optimize(" not in src
    assert "find_target_state" not in src
    # No instantiation/call of TargetStateEngine (a docstring mention is fine).
    assert "TargetStateEngine(" not in src


def test_legacy_gui_reverse_optimize_wrapper_is_removed():
    # The competing OH/NPS reverse wrapper no longer exists on the GUI service.
    assert not hasattr(svc, "reverse_optimize")
    # The canonical entry point is the sole OH/NPS reverse API on the GUI.
    assert callable(svc.reverse_optimize_canonical)


def test_find_target_state_remains_for_separate_target_state_feature():
    # The separate multi-KPI Target State path is untouched.
    assert callable(svc.find_target_state)


def test_reverse_optimize_canonical_has_no_target_state_dependency():
    src = inspect.getsource(svc.reverse_optimize_canonical)
    assert "find_target_state" not in src
    # No instantiation/call of TargetStateEngine (a docstring mention saying it
    # is NOT called is acceptable).
    assert "TargetStateEngine(" not in src
    # It routes to the canonical ReverseOptimizer engine.
    assert "ReverseOptimizer" in src


def test_gui_services_defines_no_legacy_reverse_optimize_function():
    src = inspect.getsource(svc)
    assert "def reverse_optimize(" not in src


def test_canonical_hard_bounds_preserved_in_config():
    from core.forecast_ai.config import KPI_BOUNDS
    assert KPI_BOUNDS == {
        "quality": (60, 100),
        "competency": (55, 100),
        "attendance": (65, 100),
        "release": (50, 100),
        "transfer": (0, 20),
    }
