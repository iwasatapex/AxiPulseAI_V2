import importlib

def test_run_benchmark_surface():
    module = importlib.import_module("performance.run_benchmark")
    assert hasattr(module, "benchmark")
