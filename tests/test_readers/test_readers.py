import errno
import os
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from ahcore.readers import FileImageReader, StitchingMode


# A concrete implementation of FileImageReader for testing purposes
class ConcreteFileImageReader(FileImageReader):
    def _open_file_handle(self, filename: Path) -> h5py.File:
        return h5py.File(filename, "r")

    def _read_metadata(self) -> None:
        self._metadata = {
            "mpp": self._file.attrs["mpp"],
            "tile_size": self._file.attrs["tile_size"],
            "tile_overlap": self._file.attrs["tile_overlap"],
            "size": self._file.attrs["size"],
            "num_channels": self._file.attrs["num_channels"],
            "dtype": self._file.attrs["dtype"],
            "precision": self._file.attrs["precision"],
            "multiplier": self._file.attrs["multiplier"],
            "is_binary": self._file.attrs["is_binary"],
            "has_color_profile": self._file.attrs["has_color_profile"],
            "num_tiles": self._file.attrs["num_tiles"],
        }

    def _open_file(self) -> None:
        if not self._filename.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(self._filename))

        self._file = self._open_file_handle(self._filename)

        self._read_metadata()

        if not self._metadata:
            raise ValueError("Metadata of h5 file is empty.")

        self._mpp = self._metadata["mpp"]
        self._tile_size = self._metadata["tile_size"]
        self._tile_overlap = self._metadata["tile_overlap"]
        self._size = self._metadata["size"]
        self._num_channels = self._metadata["num_channels"]
        self._dtype = self._metadata["dtype"]
        self._precision = self._metadata["precision"]
        self._multiplier = self._metadata["multiplier"]
        self._is_binary = self._metadata["is_binary"]
        self._stride = (
            self._tile_size[0] - self._tile_overlap[0],
            self._tile_size[1] - self._tile_overlap[1],
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


@pytest.fixture
def temp_h5_file(tmp_path: Path) -> Generator[Path, None, None]:
    h5_file_path = tmp_path / "test_data.h5"
    size = (350, 350)  # Total size
    mpp = 0.5
    tile_size = (200, 200)  # Size of each tile
    tile_overlap = (50, 50)  # Overlap between tiles
    num_channels = 3
    dtype = "uint8"
    precision = "FP32"
    multiplier = 1.0
    is_binary = False
    has_color_profile = False
    num_tiles = 4  # 2x2 grid of tiles

    with h5py.File(h5_file_path, "w") as f:
        f.attrs["mpp"] = mpp
        f.attrs["tile_size"] = tile_size
        f.attrs["tile_overlap"] = tile_overlap
        f.attrs["size"] = size
        f.attrs["num_channels"] = num_channels
        f.attrs["dtype"] = dtype
        f.attrs["precision"] = precision
        f.attrs["multiplier"] = multiplier
        f.attrs["is_binary"] = is_binary
        f.attrs["has_color_profile"] = has_color_profile
        f.attrs["num_tiles"] = num_tiles
        f.create_dataset("data", (num_tiles, num_channels, tile_size[0], tile_size[1]), dtype=dtype)
        f.create_dataset("tile_indices", (num_tiles,), dtype="int64")

        for i in range(num_tiles):
            f["data"][i] = np.random.randint(0, 256, (num_channels, tile_size[0], tile_size[1]), dtype=dtype)
        f["tile_indices"][:] = [0, 1, 2, 3]

    yield h5_file_path


@pytest.fixture
def sample_image_reader(temp_h5_file: Path):
    stitching_mode = StitchingMode.AVERAGE
    tile_filter = (5, 5)
    return ConcreteFileImageReader(temp_h5_file, stitching_mode, tile_filter)


def test_initialization(sample_image_reader):
    assert sample_image_reader._filename.is_file()
    assert sample_image_reader._stitching_mode == StitchingMode.AVERAGE
    assert sample_image_reader._tile_filter == (5, 5)


def test_from_file_path(temp_h5_file: Path):
    reader = ConcreteFileImageReader.from_file_path(temp_h5_file)
    assert reader._filename == temp_h5_file
    assert reader._stitching_mode == StitchingMode.AVERAGE
    assert reader._tile_filter == (5, 5)


def test_mpp(sample_image_reader):
    sample_image_reader._open_file()
    assert sample_image_reader.mpp == 0.5


def test_get_mpp(sample_image_reader):
    sample_image_reader._open_file()
    assert sample_image_reader.get_mpp(None) == 0.5
    assert sample_image_reader.get_mpp(2.0) == 0.25


def test_get_scaling(sample_image_reader):
    sample_image_reader._open_file()
    assert sample_image_reader.get_scaling(None) == 1.0
    assert sample_image_reader.get_scaling(0.25) == 2.0


def test_read_region(sample_image_reader):
    sample_image_reader._open_file()
    with patch("pyvips.Image.new_from_array") as mock_pyvips:
        # Reading a region that covers all 4 tiles
        region = sample_image_reader.read_region((0, 0), 0, (350, 350))
        mock_pyvips.assert_called_once()
        assert region is not None


def test_context_manager(sample_image_reader):
    with patch.object(sample_image_reader, "_open_file", wraps=sample_image_reader._open_file) as mock_open:
        with sample_image_reader as reader:
            mock_open.assert_called_once()
        assert reader._file is not None


def test_open_file_raises(sample_image_reader):
    with patch.object(Path, "is_file", return_value=False):
        with pytest.raises(FileNotFoundError):
            sample_image_reader._open_file()


def test_empty_tile(sample_image_reader):
    sample_image_reader._open_file()
    empty_tile = sample_image_reader._empty_tile()
    assert empty_tile.shape == (3, 200, 200)
