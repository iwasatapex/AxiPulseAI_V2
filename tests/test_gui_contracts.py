"""Regression tests for the canonical GUI contracts module.

Covers KPI hard bounds / service-boundary validation, NPS distribution
normalisation (all supported engine schemas), and the canonical
dataset-loading helper.
"""
from __future__ import annotations

import pytest

from gui import contracts as ct


# =====================================================================
# KPI bounds / validate_state (service boundary)
# =====================================================================

def _valid_state():
    return {
        "quality": 87.0,
        "competency": 93.0,
        "attendance": 90.0,
        "release": 60.0,
        "transfer": 9.0,
        "operations_health": 95.0,
        "nps": 82.0,
        "total_calls_received": 2000.0,
    }


def test_valid_state_passes():
    ct.validate_state(_valid_state())  # no exception


@pytest.mark.parametrize("key,value", [
    ("release", 49.0),   # NEVER below 50
    ("release", 49.5),
    ("release", -5.0),
    ("transfer", 21.0),  # NEVER above 20
    ("transfer", 99.0),
    ("quality", 59.0),
    ("quality", 101.0),
    ("competency", 54.0),
    ("attendance", 64.0),
    ("nps", -101.0),
    ("nps", 101.0),
])
def test_out_of_range_kpi_rejected(key, value):
    state = _valid_state()
    state[key] = value
    with pytest.raises(ValueError, match="within"):
        ct.validate_state(state)


def test_valid_boundary_values_accepted():
    # Release at exactly 50 and transfer at exactly 20 are the hard limits.
    state = _valid_state()
    state["release"] = 50.0
    state["transfer"] = 20.0
    ct.validate_state(state)


def test_unknown_keys_are_ignored():
    state = _valid_state()
    state["history_buffer"] = []
    state["custom_engine_flag"] = True
    ct.validate_state(state)  # no exception


def test_non_numeric_kpi_rejected():
    state = _valid_state()
    state["release"] = "fast"
    with pytest.raises(ValueError, match="numeric"):
        ct.validate_state(state)


# =====================================================================
# NPS distribution normalisation
# =====================================================================

def test_normalize_numeric_keys():
    assert ct.normalize_nps_distribution({5: 0.1, 6: 0.2}) == {5: 0.1, 6: 0.2}


def test_normalize_string_int_keys():
    assert ct.normalize_nps_distribution({"5": 0.1, "6": 0.2}) == {5: 0.1, 6: 0.2}


def test_normalize_prefixed_keys():
    assert ct.normalize_nps_distribution({"score_5": 0.1, "score_6": 0.2}) == {5: 0.1, 6: 0.2}


def test_normalize_mixed_keys():
    dist = ct.normalize_nps_distribution({"score_5": 0.1, 6: 0.2, "7": 0.3})
    assert dist == {5: 0.1, 6: 0.2, 7: 0.3}


def test_normalize_malformed_key_raises_clear_error():
    with pytest.raises(ValueError, match="Malformed NPS distribution key"):
        ct.normalize_nps_distribution({"nps_score": 0.5})


def test_normalize_non_mapping_raises():
    with pytest.raises(ValueError, match="must be a mapping"):
        ct.normalize_nps_distribution([1, 2, 3])


def test_normalize_non_numeric_prob_raises():
    with pytest.raises(ValueError, match="not numeric"):
        ct.normalize_nps_distribution({5: "high"})


def test_normalize_bool_key_rejected():
    with pytest.raises(ValueError, match="Malformed"):
        ct.normalize_nps_distribution({True: 0.1})


def test_normalize_empty_distribution():
    assert ct.normalize_nps_distribution({}) == {}


def test_normalize_none_raises():
    with pytest.raises(ValueError, match="must be a mapping"):
        ct.normalize_nps_distribution(None)


def test_normalize_out_of_range_score_forms():
    # score_-100 and score_85 are not valid 0..10 distribution keys.
    with pytest.raises(ValueError, match="Malformed"):
        ct.normalize_nps_distribution({"score_-100": 0.5})
    with pytest.raises(ValueError, match="Malformed"):
        ct.normalize_nps_distribution({"score_85": 0.5})


def test_normalize_score_0_valid():
    assert ct.normalize_nps_distribution({"score_0": 0.1}) == {0: 0.1}


def test_normalize_unexpected_value_type_raises():
    with pytest.raises(ValueError, match="must be a mapping"):
        ct.normalize_nps_distribution("not a dict")


def test_normalize_malformed_score_key():
    with pytest.raises(ValueError, match="Malformed"):
        ct.normalize_nps_distribution({"score_abc": 0.5})


# =====================================================================
# Dataset loading
# =====================================================================

def test_load_dataset_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("actual_quality,promoters\n80,10\n")
    df = ct.load_dataset(p)
    assert list(df.columns) == ["actual_quality", "promoters"]
    assert len(df) == 1


def test_load_dataset_json(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('[{"actual_quality": 80, "promoters": 10}]')
    df = ct.load_dataset(p)
    assert "actual_quality" in df.columns
    assert len(df) == 1


def test_load_dataset_unsupported_raises(tmp_path):
    p = tmp_path / "data.unknown"
    p.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        ct.load_dataset(p)


def test_supported_dataset_formats_are_known():
    # Core formats are always supported.
    assert {".csv", ".tsv", ".json"} <= ct.SUPPORTED_DATASET_FORMATS
    # No bogus extension is ever advertised.
    assert ".xyz" not in ct.SUPPORTED_DATASET_FORMATS


def test_trainable_formats_match_loader():
    """Anything advertised as trainable must actually load via load_dataset."""
    samples = {
        ".csv": "actual_quality,promoters\n80,10\n",
        ".tsv": "actual_quality\tpromoters\n80\t10\n",
        ".json": '[{"actual_quality": 80, "promoters": 10}]',
    }
    for ext, content in samples.items():
        assert ext in ct.SUPPORTED_DATASET_FORMATS
        with _temp_file(ext, content) as path:
            df = ct.load_dataset(path)
            assert "actual_quality" in df.columns
            assert len(df) >= 1


def test_parquet_trainable_matches_loader(tmp_path):
    if ".parquet" not in ct.SUPPORTED_DATASET_FORMATS:
        pytest.skip("parquet dependency not installed")
    import pandas as pd
    p = tmp_path / "data.parquet"
    pd.DataFrame({"actual_quality": [80.0], "promoters": [10]}).to_parquet(p)
    df = ct.load_dataset(p)
    assert "actual_quality" in df.columns
    assert len(df) == 1


def test_xlsx_trainable_matches_loader(tmp_path):
    if ".xlsx" not in ct.SUPPORTED_DATASET_FORMATS:
        pytest.skip("openpyxl dependency not installed")
    import pandas as pd
    p = tmp_path / "data.xlsx"
    pd.DataFrame({"actual_quality": [80.0], "promoters": [10]}).to_excel(p, index=False)
    df = ct.load_dataset(p)
    assert "actual_quality" in df.columns
    assert len(df) == 1


class _temp_file:
    def __init__(self, ext, content):
        import tempfile
        self._f = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        self.path = self._f.name
        with open(self.path, "w") as fh:
            fh.write(content)
        self._f.close()

    def __enter__(self):
        return self.path

    def __exit__(self, *exc):
        import os
        try:
            os.unlink(self.path)
        except OSError:
            pass
