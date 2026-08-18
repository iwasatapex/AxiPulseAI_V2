"""Regression tests: every ``st.plotly_chart`` in the GUI carries a unique key.

Streamlit auto-generates ``plotly_chart`` element IDs from the element type and
its parameters. Two identical ``st.plotly_chart(fig, width="stretch")`` calls in
one run (Predict view + its Analytics panel, both rendering the NPS score
distribution) collided with ``StreamlitDuplicateElementId``. All call sites now
pass an explicit unique ``key``.
"""
import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
GUI = ROOT / "gui"
APP = ROOT / "tests" / "gui" / "apps" / "predict_plotly_app.py"


def _plotly_chart_calls():
    """Return [(path, lineno, key)] for every ``st.plotly_chart(...)`` call."""
    calls = []
    for path in sorted(GUI.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_st = (
                isinstance(func, ast.Attribute)
                and func.attr == "plotly_chart"
                and isinstance(func.value, ast.Name)
                and func.value.id == "st"
            )
            if not is_st:
                continue
            key = None
            for kw in node.keywords:
                if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                    key = kw.value.value
            calls.append((path, node.lineno, key))
    return calls


def test_every_plotly_chart_has_a_key():
    calls = _plotly_chart_calls()
    assert calls, "no st.plotly_chart calls found in gui/"
    missing = [(str(p), line) for p, line, key in calls if key is None]
    assert not missing, f"plotly_chart call(s) without a unique key: {missing}"


def test_plotly_chart_keys_are_unique():
    calls = _plotly_chart_calls()
    seen = {}
    for path, line, key in calls:
        location = f"{path}:{line}"
        assert key not in seen, f"duplicate plotly_chart key {key!r}: {seen[key]} and {location}"
        seen[key] = location


def test_predict_view_and_analytics_do_not_collide():
    """AppTest reproduction of the reported StreamlitDuplicateElementId.

    The Predict view and its Analytics panel both render an NPS distribution
    plotly chart in the same run; with unique keys both render side by side.
    """
    at = AppTest.from_file(str(APP)).run()

    assert len(at.exception) == 0, f"streamlit run failed: {at.exception}"
    charts = at.get("plotly_chart")
    assert len(charts) == 2
