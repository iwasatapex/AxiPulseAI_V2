import pandas as pd
import pytest

from core.data import process_chunks


def test_processes_one_chunk_at_a_time():
    frames = [
        pd.DataFrame({"value": [1, 2]}),
        pd.DataFrame({"value": [3, 4]}),
        pd.DataFrame({"value": [5]}),
    ]

    seen = []

    def processor(frame):
        seen.append(len(frame))

    stats = process_chunks(frames, processor)

    assert seen == [2, 2, 1]
    assert stats.chunks == 3
    assert stats.rows == 5
    assert stats.columns == 1
    assert stats.peak_rows_per_chunk == 2


def test_processor_receives_current_frame_only():
    frames = [
        pd.DataFrame({"value": [1, 2, 3]}),
        pd.DataFrame({"value": [4]}),
    ]

    identities = []

    def processor(frame):
        identities.append(id(frame))

    stats = process_chunks(frames, processor)

    assert stats.chunks == 2
    assert len(identities) == 2
    assert identities[0] == id(frames[0])
    assert identities[1] == id(frames[1])


def test_empty_stream_is_valid():
    stats = process_chunks([], lambda frame: None)

    assert stats.chunks == 0
    assert stats.rows == 0
    assert stats.columns == 0
    assert stats.peak_rows_per_chunk == 0


def test_invalid_processor_rejected():
    with pytest.raises(TypeError):
        process_chunks([], None)


def test_invalid_chunk_rejected():
    with pytest.raises(TypeError):
        process_chunks([{"value": 1}], lambda frame: None)


def test_processing_does_not_require_concatenation():
    frames = [
        pd.DataFrame({"value": [1]}),
        pd.DataFrame({"value": [2]}),
        pd.DataFrame({"value": [3]}),
    ]

    total = 0

    def processor(frame):
        nonlocal total
        total += int(frame["value"].sum())

    stats = process_chunks(frames, processor)

    assert total == 6
    assert stats.rows == 3
