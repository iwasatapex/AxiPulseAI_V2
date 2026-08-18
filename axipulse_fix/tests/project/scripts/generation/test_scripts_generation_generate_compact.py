import importlib

def test_generate_compact_surface():
    module = importlib.import_module("scripts.generation.generate_compact")
    assert hasattr(module, "print_header")
    assert hasattr(module, "print_section")
    assert hasattr(module, "print_success")
    assert hasattr(module, "clear_screen")
    assert hasattr(module, "get_float")
    assert hasattr(module, "generate_data")
    assert hasattr(module, "display_stats")
    assert hasattr(module, "save_files")
    assert hasattr(module, "main")
    assert hasattr(module, "get_user_inputs")
    assert hasattr(module, "get_mode")
    assert hasattr(module, "Colors")
    assert hasattr(module, "CompactGenerator")
