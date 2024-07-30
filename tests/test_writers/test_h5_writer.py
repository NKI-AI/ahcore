import json
from pathlib import Path
from typing import Any, Generator

import h5py
import numpy as np
import pytest

from ahcore.utils.types import GenericNumberArray, InferencePrecision
from ahcore.writers import H5FileImageWriter


@pytest.fixture
def temp_h5_file(tmp_path: Path) -> Generator[Path, None, None]:
    h5_file_path = tmp_path / "test_data.h5"
    yield h5_file_path
    if h5_file_path.exists():
        h5_file_path.unlink()


@pytest.fixture
def dummy_batch_data() -> Generator[tuple[GenericNumberArray, GenericNumberArray], None, None]:
    dummy_coordinates = np.array([[0, 0]])
    dummy_batch = np.random.rand(1, 3, 200, 200).astype(np.float32)
    yield dummy_coordinates, dummy_batch


@pytest.fixture
def dummy_batch_generator(
    dummy_batch_data: tuple[GenericNumberArray, GenericNumberArray]
) -> Generator[tuple[GenericNumberArray, GenericNumberArray], None, None]:
    def generator() -> Generator[tuple[GenericNumberArray, GenericNumberArray], None, None]:
        yield dummy_batch_data

    return generator


def test_h5_file_image_writer_creation(temp_h5_file: Path) -> None:
    size = (200, 200)
    mpp = 0.5
    tile_size = (200, 200)
    tile_overlap = (0, 0)
    num_samples = 1

    writer = H5FileImageWriter(
        filename=temp_h5_file,
        size=size,
        mpp=mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        num_samples=num_samples,
    )

    assert writer._filename == temp_h5_file
    assert writer._size == size
    assert writer._mpp == mpp
    assert writer._tile_size == tile_size
    assert writer._tile_overlap == tile_overlap
    assert writer._num_samples == num_samples


def test_h5_file_image_writer_consume(temp_h5_file: Path, dummy_batch_generator: Any) -> None:
    size = (200, 200)
    mpp = 0.5
    tile_size = (200, 200)
    tile_overlap = (0, 0)
    num_samples = 1

    writer = H5FileImageWriter(
        filename=temp_h5_file,
        size=size,
        mpp=mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        num_samples=num_samples,
    )

    writer.consume(dummy_batch_generator())

    with h5py.File(temp_h5_file, "r") as h5file:
        assert "data" in h5file
        assert "coordinates" in h5file
        assert np.array(h5file["data"]).shape == (num_samples, 3, 200, 200)
        assert np.array(h5file["coordinates"]).shape == (num_samples, 2)

        dummy_coordinates, dummy_batch = next(dummy_batch_generator())
        assert np.allclose(h5file["data"], dummy_batch)
        assert np.allclose(h5file["coordinates"], dummy_coordinates)


def test_h5_file_image_writer_metadata(temp_h5_file: Path, dummy_batch_generator: Any) -> None:
    size = (200, 200)
    mpp = 0.5
    tile_size = (200, 200)
    tile_overlap = (0, 0)
    num_samples = 1
    num_tiles = 1
    grid_order = "C"
    tiling_mode = "overflow"
    format = "RAW"
    dtype = "float32"
    is_binary = False
    grid_offset = (0, 0)
    precision = InferencePrecision.FP32
    multiplier = 1.0

    writer = H5FileImageWriter(
        filename=temp_h5_file,
        size=size,
        mpp=mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        num_samples=num_samples,
        precision=precision,
    )

    writer.consume(dummy_batch_generator())

    with h5py.File(temp_h5_file, "r") as h5file:
        metadata = json.loads(h5file.attrs["metadata"])
        assert metadata["size"] == list(size)
        assert metadata["mpp"] == mpp
        assert metadata["num_samples"] == num_samples
        assert metadata["tile_size"] == list(tile_size)
        assert metadata["tile_overlap"] == list(tile_overlap)
        assert metadata["num_tiles"] == num_tiles
        assert metadata["grid_order"] == grid_order
        assert metadata["tiling_mode"] == tiling_mode
        assert metadata["format"] == format
        assert metadata["dtype"] == dtype
        assert metadata["is_binary"] == is_binary
        assert metadata["grid_offset"] == list(grid_offset)
        assert metadata["precision"] == precision
        assert metadata["multiplier"] == multiplier


def test_h5_file_image_writer_multiple_tiles(temp_h5_file: Path) -> None:
    size = (400, 400)
    mpp = 0.5
    tile_size = (200, 200)
    tile_overlap = (0, 0)
    num_samples = 2

    writer = H5FileImageWriter(
        filename=temp_h5_file,
        size=size,
        mpp=mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        num_samples=num_samples,
    )

    def multiple_tile_generator() -> Generator[tuple[GenericNumberArray, GenericNumberArray], None, None]:
        for i in range(num_samples):
            coordinates = np.array([[i * 200, 0]])
            batch = np.random.rand(1, 3, 200, 200).astype(np.float32)
            yield coordinates, batch

    writer.consume(multiple_tile_generator())

    with h5py.File(temp_h5_file, "r") as h5file:
        assert "data" in h5file
        assert "coordinates" in h5file
        assert h5file["data"].shape == (num_samples, 3, 200, 200)
        assert h5file["coordinates"].shape == (num_samples, 2)
        for i in range(num_samples):
            assert np.allclose(h5file["coordinates"][i], [i * 200, 0])
