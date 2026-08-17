import importlib

def test_commands_display_surface():
    module = importlib.import_module("cli.commands_display")
    assert hasattr(module, "render_full_dashboard")
    assert hasattr(module, "cprint")
    assert hasattr(module, "left_print")
