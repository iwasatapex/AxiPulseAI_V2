from pathlib import Path


def test_benchmark_exists():

    assert Path(
        "performance/run_benchmark.py"
    ).exists()
