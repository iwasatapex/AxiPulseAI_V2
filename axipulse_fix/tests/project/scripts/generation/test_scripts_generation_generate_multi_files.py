import importlib

def test_generate_multi_files_surface():
    module = importlib.import_module("scripts.generation.generate_multi_files")
    assert hasattr(module, "print_header")
    assert hasattr(module, "print_section")
    assert hasattr(module, "print_success")
    assert hasattr(module, "main")
    assert hasattr(module, "get_user_inputs")
    assert hasattr(module, "get_float_input")
    assert hasattr(module, "generate_data")
    assert hasattr(module, "save_files")
    assert hasattr(module, "show_summary")
    assert hasattr(module, "run")
    assert hasattr(module, "Colors")
    assert hasattr(module, "MultiFileGenerator")
