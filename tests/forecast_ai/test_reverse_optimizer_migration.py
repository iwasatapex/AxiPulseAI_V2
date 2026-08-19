"""Regression tests for the reverse-optimizer architecture migration.

Proves:
  - The CLI (AxisPulseAI) no longer exposes the legacy Target State Engine path.
  - ReverseOptimizer does NOT depend on / import TargetStateEngine.
  - The universal probabilistic infrastructure is no longer coupled to
    TargetStateEngine (no adapt_target_state_prediction).
"""
import importlib
import inspect


def test_axispulseai_no_longer_exposes_target_state_engine():
    """The user-facing Target State Engine path must be gone from the CLI."""
    module = importlib.import_module("AxisPulseAI")
    assert not hasattr(module, "do_target_state")
    assert not hasattr(module, "TargetStateEngine")


def test_axispulseai_does_not_import_target_state_engine():
    """AxisPulseAI must not contain any reference to TargetStateEngine or the
    target_state_engine package (dead/forbidden dependency)."""
    import AxisPulseAI as cli

    src = inspect.getsource(cli)
    assert "TargetStateEngine" not in src
    assert "target_state_engine" not in src
    assert "do_target_state" not in src


def test_reverse_optimizer_does_not_depend_on_target_state_engine():
    """ReverseOptimizer must generate/evaluate candidate states via the
    canonical PredictionService and must never import or call TargetStateEngine."""
    import core.forecast_ai.optimization.optimizer as optmod

    src = inspect.getsource(optmod.ReverseOptimizer)
    assert "TargetStateEngine" not in src
    assert "target_state_engine" not in src
    # It builds candidates itself; it does not delegate to any target-state scan.
    assert "find_target_state" not in src


def test_universal_probabilistic_infra_not_coupled_to_target_state_engine():
    """The domain-neutral probabilistic adapters must not expose a
    TargetStateEngine-specific adapter."""
    from core.probabilistic import domain_adapters
    from core.probabilistic import __all__ as prob_all

    assert not hasattr(domain_adapters, "adapt_target_state_prediction")
    assert "adapt_target_state_prediction" not in (domain_adapters.__all__ or [])
    assert "adapt_target_state_prediction" not in prob_all


def test_target_state_engine_init_no_longer_imports_removed_adapter():
    """core.target_state_engine/__init__ must not reference the removed
    probabilistic adapter."""
    module = importlib.import_module("core.target_state_engine")
    src = inspect.getsource(module)
    assert "adapt_target_state_prediction" not in src
