from pathlib import Path

import pandas as pd
import pytest

from core.data import CSVChunk, iter_csv_chunks


def test_csv_streaming_reads_multiple_chunks(tmp_path: Path):
    path = tmp_path / "large.csv"

    frame = pd.DataFrame(
        {
            "value": list(range(10)),
            "category": ["A", "B"] * 5,
        }
    )

    frame.to_csv(path, index=False)

    chunks = list(
        iter_csv_chunks(
            path,
            chunksize=3,
        )
    )

    assert len(chunks) == 4
    assert all(isinstance(chunk, CSVChunk) for chunk in chunks)

    assert [chunk.chunk_index for chunk in chunks] == [
        0,
        1,
        2,
        3,
    ]

    assert [chunk.rows for chunk in chunks] == [
        3,
        3,
        3,
        1,
    ]

    reconstructed = pd.concat(
        [chunk.data for chunk in chunks],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(
        reconstructed,
        frame,
    )


def test_csv_streaming_preserves_source(tmp_path: Path):
    path = tmp_path / "source.csv"

    frame = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    frame.to_csv(path, index=False)

    before = path.read_bytes()

    chunks = list(
        iter_csv_chunks(
            path,
            chunksize=2,
        )
    )

    assert len(chunks) == 2
    assert path.read_bytes() == before


def test_csv_streaming_rejects_invalid_chunksize(tmp_path: Path):
    path = tmp_path / "source.csv"

    pd.DataFrame({"value": [1]}).to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError):
        list(
            iter_csv_chunks(
                path,
                chunksize=0,
            )
        )


def test_csv_streaming_missing_file():
    with pytest.raises(FileNotFoundError):
        list(
            iter_csv_chunks(
                "/does/not/exist.csv",
                chunksize=2,
            )
        )


def test_chunk_metadata_is_consistent(tmp_path: Path):
    path = tmp_path / "source.csv"

    pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
        }
    ).to_csv(
        path,
        index=False,
    )

    chunks = list(
        iter_csv_chunks(
            path,
            chunksize=2,
        )
    )

    for chunk in chunks:
        assert chunk.rows == len(chunk.data)
        assert chunk.source == str(path)
