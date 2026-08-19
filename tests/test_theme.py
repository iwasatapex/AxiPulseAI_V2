"""Focused tests for the AxiPulseAI GUI theme system and header layout fix.

Covers:
  1. exactly 10 themes exist
  2. exactly 3 themes are bright
  3. exactly 7 themes are dark
  4. theme names are unique
  5. theme selection persists through Streamlit reruns (session state)
  6. CSS/theme variables are generated (--ax-*)
  7. default theme is valid
  8. header CSS/layout no longer uses the broken clipping behavior
  9. every theme defines all required semantic keys with valid hex colors
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.theme import (  # noqa: E402
    BRIGHT_THEMES,
    DARK_THEMES,
    DEFAULT_THEME,
    REQUIRED_KEYS,
    THEMES,
    css_variables,
    get_theme,
    is_bright,
    is_dark,
    theme_names,
)


# ---------------------------------------------------------------------------
# Theme inventory
# ---------------------------------------------------------------------------

def test_exactly_10_themes_exist():
    assert len(THEMES) == 10


def test_exactly_3_bright_themes():
    assert len([n for n in THEMES if is_bright(n)]) == 3
    assert set(BRIGHT_THEMES) == {"Bright", "Arctic", "Ivory"}


def test_exactly_7_dark_themes():
    assert len([n for n in THEMES if is_dark(n)]) == 7
    assert set(DARK_THEMES) == {
        "Midnight", "Graphite", "Deep Ocean", "Purple Night",
        "Emerald Night", "Carbon", "Cyber Dark",
    }


def test_theme_names_are_unique():
    assert len(set(THEMES.keys())) == len(THEMES) == len(theme_names())


def test_default_theme_is_valid_and_dark():
    assert DEFAULT_THEME in THEMES
    assert THEMES[DEFAULT_THEME]["mode"] == "dark"


def test_every_theme_defines_all_semantic_keys():
    for name, theme in THEMES.items():
        for key in REQUIRED_KEYS:
            assert key in theme, f"{name} missing key {key}"


def test_theme_colors_are_valid_hex():
    hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
    color_keys = [k for k in REQUIRED_KEYS if k != "mode"]
    for name, theme in THEMES.items():
        for key in color_keys:
            assert hex_re.match(theme[key]), \
                f"{name}.{key} is not a #rrggbb hex: {theme[key]!r}"


def test_bright_themes_are_light_and_dark_themes_are_dark():
    for n in BRIGHT_THEMES:
        assert THEMES[n]["mode"] == "light"
    for n in DARK_THEMES:
        assert THEMES[n]["mode"] == "dark"


# ---------------------------------------------------------------------------
# CSS variables generation
# ---------------------------------------------------------------------------

def test_css_variables_are_generated_for_each_theme():
    for name in THEMES:
        css = css_variables(name)
        assert ":root" in css
        assert "--ax-bg" in css
        assert "--ax-surface" in css
        assert "--ax-text" in css
        assert "--ax-accent" in css
        assert "--ax-muted" in css
        assert "--ax-border" in css


def test_css_variables_reflect_theme_values():
    css = css_variables("Midnight")
    assert "--ax-bg: #0e1117" in css
    css_bright = css_variables("Bright")
    assert "--ax-bg: #f8fafc" in css_bright


def test_get_theme_falls_back_to_default_on_unknown():
    assert get_theme("Does Not Exist") is THEMES[DEFAULT_THEME]
    assert get_theme("Graphite") is THEMES["Graphite"]


# ---------------------------------------------------------------------------
# Header layout fix (no known broken clipping behavior)
# ---------------------------------------------------------------------------

def test_header_css_has_sufficient_top_padding():
    """The block container must reserve top clearance so the header is never
    clipped/undercut by the Streamlit toolbar."""
    import inspect
    from gui import components

    src = inspect.getsource(components.apply_theme)
    # The old broken behavior was a tiny 1.6rem top padding; the fix uses
    # generous but bounded top clearance plus vertical header breathing room.
    assert "padding-top: 2.6rem" in src
    assert "min-height: 56px" in src
    assert "line-height: 1.4" in src
    assert "overflow: visible" in src
    # No absolute-pixel fixed positioning / brittle offsets in the header.
    assert "position: fixed" not in src
    assert "position: sticky" not in src


def test_theme_selector_persists_through_reruns_and_reapplies(monkeypatch):
    """Changing the theme via the sidebar persists it in session state and the
    re-applied CSS reflects the new theme."""
    from streamlit.testing.v1 import AppTest

    app_script = ROOT / "tests" / "rev_theme_app.py"
    at = AppTest.from_file(str(app_script)).run()
    assert not list(at.exception), list(at.exception)

    # Default theme applied up front (no persisted value yet).
    assert "apgui_theme" not in at.session_state
    assert "--ax-bg: #0e1117" in at.markdown[0].value  # Midnight default

    # Switch to a bright theme and confirm the stored value + re-applied CSS.
    at.selectbox[0].set_value("Arctic").run()
    assert at.session_state["apgui_theme"] == "Arctic"
    assert "--ax-bg: #eef4fb" in at.markdown[0].value

    # Switch again — the stored value must update, not reset to default.
    at.selectbox[0].set_value("Cyber Dark").run()
    assert at.session_state["apgui_theme"] == "Cyber Dark"
    assert "--ax-bg: #0a0a12" in at.markdown[0].value
