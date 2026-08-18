from pathlib import Path


def test_load_test_exists():

    assert Path(
        "load_tests/locustfile.py"
    ).exists()
