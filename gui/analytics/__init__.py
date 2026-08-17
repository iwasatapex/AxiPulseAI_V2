"""AxiPulseAI V2 GUI analytics layer.

Consumes canonical engine outputs and stored session results to produce
diagnostic structure for every major GUI feature. No simulator / model
mathematics lives here and no process-global analytics state is created.

Exposes ``analytics.<feature>.render_analytics(st, ...)`` for the views.
The pure analysis functions are unit-tested in ``tests/test_gui_analytics.py``.
"""
from gui.analytics import common  # noqa: F401
