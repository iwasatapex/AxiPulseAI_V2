"""Centralized theme configuration for the AxiPulseAI GUI.

There are exactly 10 themes: 3 bright + 7 dark. Each theme defines the major
semantic colors the whole GUI consumes through root-level CSS variables
(``--ax-*``). No page scatters hard-coded colors: components reference these
variables, and this module is the single source of truth.

Themes:
    Bright     (light)   clean modern light UI, white/light-gray, blue/purple accent
    Arctic     (light)   cool white/light-blue, crisp blue/cyan accents
    Ivory      (light)   warm off-white/cream, dark charcoal text, gold/purple accent
    Midnight   (dark)    near-black/navy, blue/purple accent (default = current look)
    Graphite   (dark)    charcoal/graphite surfaces, restrained blue accent
    Deep Ocean (dark)    very dark blue, cyan/blue accent
    Purple Night (dark)  deep violet/navy, purple accent
    Emerald Night (dark) near-black green, emerald accent
    Carbon     (dark)    black/charcoal, white/gray text, minimal accent
    Cyber Dark (dark)    near-black, cyan/purple neon accents
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Theme classification
# ---------------------------------------------------------------------------

BRIGHT_THEMES = ("Bright", "Arctic", "Ivory")
DARK_THEMES = (
    "Midnight",
    "Graphite",
    "Deep Ocean",
    "Purple Night",
    "Emerald Night",
    "Carbon",
    "Cyber Dark",
)

# The default matches the current Axipulse dark appearance as closely as
# possible so switching to the selector is visually seamless.
DEFAULT_THEME = "Midnight"

_THEME_ORDER = BRIGHT_THEMES + DARK_THEMES

# Semantic keys every theme must define.
REQUIRED_KEYS = (
    "mode",            # "light" | "dark"
    "bg",              # page background
    "surface",         # card / surface background
    "surface_2",       # secondary surface (metrics, inputs' siblings)
    "text",            # primary text
    "muted",           # muted / secondary text
    "border",          # borders / dividers
    "accent",          # primary accent
    "accent_2",        # secondary accent
    "success",         # success / positive
    "warning",         # warning / degraded
    "error",           # error / danger
    "info",            # informational
    "input_bg",        # text/number input background
    "sidebar_bg",      # sidebar background
    "header_bg",       # app header / block background
    "nav_selected",    # selected navigation / active highlight
    "logo_text",       # text inside the logo badge
)


def _t(mode: str, **kw) -> Dict[str, str]:
    """Build a theme dict from keyword values + validate required keys."""
    theme = {"mode": mode}
    theme.update({k: v for k, v in kw.items() if k in REQUIRED_KEYS})
    missing = [k for k in REQUIRED_KEYS if k not in theme]
    if missing:
        raise ValueError(f"Theme missing required keys: {missing}")
    return theme


THEMES: Dict[str, Dict[str, str]] = {
    # ---------------------------------------------------------- bright (3)
    "Bright": _t(
        mode="light",
        bg="#f8fafc", surface="#ffffff", surface_2="#f1f5f9",
        text="#0f172a", muted="#64748b", border="#e2e8f0",
        accent="#6366f1", accent_2="#0ea5e9",
        success="#16a34a", warning="#d97706", error="#dc2626", info="#2563eb",
        input_bg="#ffffff", sidebar_bg="#ffffff", header_bg="#ffffff",
        nav_selected="#eef2ff", logo_text="#ffffff",
    ),
    "Arctic": _t(
        mode="light",
        bg="#eef4fb", surface="#ffffff", surface_2="#e3eefb",
        text="#0b2438", muted="#5b7a96", border="#cfdff0",
        accent="#0284c7", accent_2="#06b6d4",
        success="#059669", warning="#d97706", error="#dc2626", info="#2563eb",
        input_bg="#ffffff", sidebar_bg="#e9f2fb", header_bg="#ffffff",
        nav_selected="#dbeafe", logo_text="#ffffff",
    ),
    "Ivory": _t(
        mode="light",
        bg="#faf6ee", surface="#fffdf7", surface_2="#f3ecdd",
        text="#1f2937", muted="#8a7f6b", border="#e7ddc8",
        accent="#7c3aed", accent_2="#c59a2a",
        success="#4d7c0f", warning="#b45309", error="#b91c1c", info="#2563eb",
        input_bg="#fffdf7", sidebar_bg="#f6efe0", header_bg="#fffdf7",
        nav_selected="#f1e7d1", logo_text="#ffffff",
    ),
    # ----------------------------------------------------------- dark (7)
    "Midnight": _t(
        mode="dark",
        bg="#0e1117", surface="#161a22", surface_2="#1f2430",
        text="#fafafa", muted="#94a3b8", border="#2a2f3a",
        accent="#6366f1", accent_2="#22d3ee",
        success="#22c55e", warning="#f59e0b", error="#ef4444", info="#38bdf8",
        input_bg="#1a1f2a", sidebar_bg="#0e1117", header_bg="#0e1117",
        nav_selected="#2b3050", logo_text="#ffffff",
    ),
    "Graphite": _t(
        mode="dark",
        bg="#171717", surface="#1f1f1f", surface_2="#2a2a2a",
        text="#e5e7eb", muted="#9ca3af", border="#3a3a3a",
        accent="#60a5fa", accent_2="#94a3b8",
        success="#34d399", warning="#fbbf24", error="#f87171", info="#60a5fa",
        input_bg="#262626", sidebar_bg="#1a1a1a", header_bg="#171717",
        nav_selected="#2b2b2b", logo_text="#ffffff",
    ),
    "Deep Ocean": _t(
        mode="dark",
        bg="#081120", surface="#0f1b30", surface_2="#16263f",
        text="#e2edff", muted="#7f9cc4", border="#1f3450",
        accent="#22d3ee", accent_2="#3b82f6",
        success="#34d399", warning="#fbbf24", error="#f87171", info="#38bdf8",
        input_bg="#0f1b30", sidebar_bg="#0a1424", header_bg="#081120",
        nav_selected="#1b3a5f", logo_text="#0b1220",
    ),
    "Purple Night": _t(
        mode="dark",
        bg="#140b24", surface="#1d1230", surface_2="#281a42",
        text="#f1e9ff", muted="#a894d0", border="#3a2a5c",
        accent="#a78bfa", accent_2="#22d3ee",
        success="#34d399", warning="#fbbf24", error="#f87171", info="#818cf8",
        input_bg="#1d1230", sidebar_bg="#160d28", header_bg="#140b24",
        nav_selected="#3b2b66", logo_text="#1a1030",
    ),
    "Emerald Night": _t(
        mode="dark",
        bg="#07130d", surface="#0d1f16", surface_2="#142b20",
        text="#e6f7ef", muted="#7fb8a1", border="#1d3a2c",
        accent="#34d399", accent_2="#2dd4bf",
        success="#34d399", warning="#fbbf24", error="#f87171", info="#5eead4",
        input_bg="#0d1f16", sidebar_bg="#091710", header_bg="#07130d",
        nav_selected="#1e4d39", logo_text="#06281d",
    ),
    "Carbon": _t(
        mode="dark",
        bg="#0a0a0a", surface="#121212", surface_2="#1c1c1c",
        text="#f5f5f5", muted="#9ca3af", border="#2e2e2e",
        accent="#71717a", accent_2="#9ca3af",
        success="#4ade80", warning="#facc15", error="#f87171", info="#a1a1aa",
        input_bg="#161616", sidebar_bg="#0d0d0d", header_bg="#0a0a0a",
        nav_selected="#2a2a2a", logo_text="#ffffff",
    ),
    "Cyber Dark": _t(
        mode="dark",
        bg="#0a0a12", surface="#12121d", surface_2="#1c1c2c",
        text="#e6e6f0", muted="#8b8ba3", border="#2c2c44",
        accent="#22d3ee", accent_2="#a855f7",
        success="#34d399", warning="#fbbf24", error="#f87171", info="#67e8f9",
        input_bg="#141420", sidebar_bg="#0d0d16", header_bg="#0a0a12",
        nav_selected="#312e81", logo_text="#0a0a12",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def theme_names() -> List[str]:
    """Return all theme names (bright first, then dark) in display order."""
    return list(_THEME_ORDER)


def get_theme(name) -> Dict[str, str]:
    """Return a theme dict by name (falls back to the default on unknown)."""
    if name in THEMES:
        return THEMES[name]
    return THEMES[DEFAULT_THEME]


def is_bright(name) -> bool:
    """True for one of the 3 bright themes."""
    return name in BRIGHT_THEMES


def is_dark(name) -> bool:
    """True for one of the 7 dark themes."""
    return name in DARK_THEMES


def css_variables(theme) -> str:
    """Return the ``:root { --ax-... }`` block for a theme (dict or name)."""
    if isinstance(theme, str):
        theme = get_theme(theme)
    lines = []
    order = (
        ("bg", "bg"), ("surface", "surface"), ("surface_2", "surface-2"),
        ("text", "text"), ("muted", "muted"), ("border", "border"),
        ("accent", "accent"), ("accent_2", "accent-2"),
        ("success", "success"), ("warning", "warning"), ("error", "error"),
        ("info", "info"), ("input_bg", "input-bg"),
        ("sidebar_bg", "sidebar-bg"), ("header_bg", "header-bg"),
        ("nav_selected", "nav-selected"), ("logo_text", "logo-text"),
    )
    for key, var in order:
        lines.append(f"    --ax-{var}: {theme[key]};")
    lines.append("    --ax-radius: 12px;")
    lines.append("    --ax-radius-sm: 8px;")
    return ":root {\n" + "\n".join(lines) + "\n}"
