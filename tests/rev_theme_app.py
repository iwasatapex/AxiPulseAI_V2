"""Headless Streamlit harness for the theme selector (AppTest).

Driven by streamlit.testing.v1.AppTest; not collected by pytest (does not match
``test_*.py``). Renders the real ``gui.app.main()`` so the theme selector and
the re-applied ``apply_theme`` CSS can be exercised end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gui.app as app  # noqa: E402

app.main()
