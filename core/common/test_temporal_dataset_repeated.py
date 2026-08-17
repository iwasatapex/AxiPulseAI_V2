"""
Focused tests for shift_target_next_day() repeated-date alignment fix.
Run: python3 -m pytest core/common/test_temporal_dataset_repeated.py -v
"""
import pandas as pd
import pytest

from core.common.temporal_dataset import shift_target_next_day


def _dates(*specs):
    """specs like '2024-01-01' repeated n times via list flattening."""
    return pd.to_datetime(specs)


# ---------------------------------------------------------------------
# 1. Unique daily rows (ordinary path, must behave exactly as before)
# ---------------------------------------------------------------------
def test_unique_daily_rows_series():
    times = _dates("2024-01-01", "2024-01-02", "2024-01-03")
    target = pd.Series([10, 20, 30])

    shifted, target_times = shift_target_next_day(target, times)

    assert shifted.tolist()[:2] == [20, 30]
    assert pd.isna(shifted.iloc[2])
    assert target_times.iloc[0] == pd.Timestamp("2024-01-02")
    assert target_times.iloc[1] == pd.Timestamp("2024-01-03")
    assert pd.isna(target_times.iloc[2])


# ---------------------------------------------------------------------
# 2. Repeated dates with EQUAL sample counts per date
# ---------------------------------------------------------------------
def test_repeated_dates_equal_counts():
    times = _dates(
        "2024-01-01", "2024-01-01", "2024-01-01",
        "2024-01-02", "2024-01-02", "2024-01-02",
    )
    # value = 100*date_idx + sample_idx, so we can verify exact pairing
    target = pd.Series([0, 1, 2, 100, 101, 102])

    shifted, target_times = shift_target_next_day(target, times)

    # Jan1/sample0 -> Jan2/sample0 (value 100), etc.
    assert shifted.tolist()[:3] == [100, 101, 102]
    # Jan2 rows have no next date -> missing
    assert shifted.iloc[3:].isna().all()
    assert (target_times.iloc[:3] == pd.Timestamp("2024-01-02")).all()
    assert target_times.iloc[3:].isna().all()


# ---------------------------------------------------------------------
# 3. Repeated dates with UNEQUAL sample counts
# ---------------------------------------------------------------------
def test_repeated_dates_unequal_counts():
    # Jan1 has 3 samples (idx 0,1,2); Jan2 has only 2 samples (idx 0,1);
    # Jan3 has 3 samples (idx 0,1,2).
    times = _dates(
        "2024-01-01", "2024-01-01", "2024-01-01",
        "2024-01-02", "2024-01-02",
        "2024-01-03", "2024-01-03", "2024-01-03",
    )
    target = pd.Series([0, 1, 2, 10, 11, 20, 21, 22])

    shifted, target_times = shift_target_next_day(target, times)

    # Jan1/sample0 -> Jan2/sample0 (10)
    assert shifted.iloc[0] == 10
    # Jan1/sample1 -> Jan2/sample1 (11)
    assert shifted.iloc[1] == 11
    # Jan1/sample2 -> Jan2 has NO sample index 2 -> missing (must NOT
    # borrow Jan3's sample2, since Jan2 is the immediate next date)
    assert pd.isna(shifted.iloc[2])
    assert pd.isna(target_times.iloc[2])

    # Jan2/sample0 -> Jan3/sample0 (20)
    assert shifted.iloc[3] == 20
    # Jan2/sample1 -> Jan3/sample1 (21)
    assert shifted.iloc[4] == 21

    # Jan3 rows: no next date at all -> all missing
    assert shifted.iloc[5:].isna().all()


# ---------------------------------------------------------------------
# 4. Final date has no future target
# ---------------------------------------------------------------------
def test_final_date_no_future_target():
    times = _dates("2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02")
    target = pd.Series([1, 2, 3, 4])

    shifted, target_times = shift_target_next_day(target, times)

    # last date's rows (index 2, 3) must have no target
    assert pd.isna(shifted.iloc[2])
    assert pd.isna(shifted.iloc[3])
    assert pd.isna(target_times.iloc[2])
    assert pd.isna(target_times.iloc[3])


# ---------------------------------------------------------------------
# 5. target_time strictly greater than prediction_cutoff for every valid row
# ---------------------------------------------------------------------
def test_target_time_strictly_after_cutoff_repeated():
    times = _dates(
        "2024-01-01", "2024-01-01",
        "2024-01-02", "2024-01-02",
        "2024-01-03", "2024-01-03",
    )
    target = pd.Series([1, 2, 3, 4, 5, 6])

    shifted, target_times = shift_target_next_day(target, times)

    valid = target_times.notna()
    for cutoff, ttime in zip(times[valid], target_times[valid]):
        assert ttime > cutoff


def test_temporal_violation_still_raises():
    # Same date repeated with target.shift(-1) semantics would previously
    # pair same-date rows -> contract violation. Confirm the OLD unsafe
    # behavior (bypassing the fix) is exactly what the contract check
    # would reject, by directly invoking assert_forecast_boundary.
    from core.common.temporal_contract import assert_forecast_boundary

    same_day = pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="Temporal contract violation"):
        assert_forecast_boundary(same_day, same_day)

    with pytest.raises(ValueError, match="Temporal contract violation"):
        assert_forecast_boundary(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01"))


# ---------------------------------------------------------------------
# 6. Series target (repeated dates)
# ---------------------------------------------------------------------
def test_series_target_repeated_dates():
    times = _dates("2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02")
    target = pd.Series([5.0, 6.0, 7.0, 8.0])

    shifted, target_times = shift_target_next_day(target, times, field_name="nps")

    assert isinstance(shifted, pd.Series)
    assert shifted.iloc[0] == 7.0
    assert shifted.iloc[1] == 8.0


# ---------------------------------------------------------------------
# 7. DataFrame 11-bucket NPS target (repeated dates, unequal counts)
# ---------------------------------------------------------------------
def test_dataframe_11_bucket_target_repeated_dates():
    times = _dates(
        "2024-01-01", "2024-01-01", "2024-01-01",
        "2024-01-02", "2024-01-02",
    )
    cols = [f"score_{i}" for i in range(11)]
    rows = []
    for d in range(2):
        for s in range(3 if d == 0 else 2):
            rows.append([d * 100 + s] * 11)
    target = pd.DataFrame(rows, columns=cols)

    shifted, target_times = shift_target_next_day(target, times, field_name="NPS score distribution")

    assert isinstance(shifted, pd.DataFrame)
    assert list(shifted.columns) == cols

    # Jan1/sample0 -> Jan2/sample0 (all-100 row)
    assert (shifted.iloc[0] == 100).all()
    # Jan1/sample1 -> Jan2/sample1 (all-101 row)
    assert (shifted.iloc[1] == 101).all()
    # Jan1/sample2 -> Jan2 has no sample_idx 2 -> missing
    assert shifted.iloc[2].isna().all()
    # Jan2 rows: no next date -> missing
    assert shifted.iloc[3].isna().all()
    assert shifted.iloc[4].isna().all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
