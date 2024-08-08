import errno
import os
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from ahcore.readers import FileImageReader, StitchingMode
from ahcore.utils.types import GenericNumberArray


def colorize(image_array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    color_map = {
        0: np.array([255, 0, 0]),
        1: np.array([0, 255, 0]),
        2: np.array([0, 0, 255]),
        3: np.array([0, 255, 255]),
    }

    colored_array = np.zeros((*image_array.shape, 3), dtype=np.uint8)

    # Apply the color map
    for class_index, color in color_map.items():
        mask = image_array == class_index
        # Use numpy broadcasting to set the color
        colored_array[mask] = color

    return colored_array


def create_colored_tiles(
    tile_size: tuple[int, int],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    # Create class index images
    red_image = np.full((tile_size[0], tile_size[1]), 2, dtype=np.uint8)
    green_image = np.full((tile_size[0], tile_size[1]), 1, dtype=np.uint8)
    blue_image = np.full((tile_size[0], tile_size[1]), 0, dtype=np.uint8)
    yellow_image = np.full((tile_size[0], tile_size[1]), 3, dtype=np.uint8)

    # Colorize the images
    red_colored = colorize(red_image)
    green_colored = colorize(green_image)
    blue_colored = colorize(blue_image)
    yellow_colored = colorize(yellow_image)
    return red_colored, green_colored, blue_colored, yellow_colored


class TestFileImageReader(FileImageReader):
    def _open_file_handle(self, filename: Path) -> h5py.File:
        return h5py.File(filename, "r")

    def _read_metadata(self) -> None:
        self._metadata = {  # type: ignore
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
        # Initializations only for mypy
        self.__empty_tile: GenericNumberArray | None = None

        self._file: Any
        self._metadata = None
        self._mpp = None
        self._tile_size = None
        self._tile_overlap = None
        self._size = None
        self._num_channels = None
        self._dtype = None
        self._stride = None
        self._precision = None
        self._multiplier = None
        self._is_binary = None

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
        f.create_dataset("tile_indices", (num_tiles,), dtype=np.uint8)

        red_colored, green_colored, blue_colored, yellow_colored = create_colored_tiles(tile_size)

        # Transpose to match the expected shape (C, H, W)
        red_array = red_colored.transpose(2, 0, 1)
        green_array = green_colored.transpose(2, 0, 1)
        blue_array = blue_colored.transpose(2, 0, 1)
        yellow_array = yellow_colored.transpose(2, 0, 1)

        f["data"][0] = red_array
        f["data"][1] = green_array
        f["data"][2] = blue_array
        f["data"][3] = yellow_array
        f["tile_indices"][:] = [0, 1, 2, 3]

    yield h5_file_path


@pytest.fixture
def sample_image_reader(temp_h5_file: Path) -> TestFileImageReader:
    stitching_mode = StitchingMode.AVERAGE
    return TestFileImageReader(temp_h5_file, stitching_mode)


def test_initialization(sample_image_reader: TestFileImageReader) -> None:
    assert sample_image_reader._filename.is_file()
    assert sample_image_reader._stitching_mode == StitchingMode.AVERAGE


def test_from_file_path(temp_h5_file: Path) -> None:
    reader = TestFileImageReader.from_file_path(temp_h5_file)
    assert reader._filename == temp_h5_file
    assert reader._stitching_mode == StitchingMode.CROP


def test_mpp(sample_image_reader: TestFileImageReader) -> None:
    sample_image_reader._open_file()
    assert sample_image_reader.mpp == 0.5


def test_get_mpp(sample_image_reader: TestFileImageReader) -> None:
    sample_image_reader._open_file()
    assert sample_image_reader.get_mpp(None) == 0.5
    assert sample_image_reader.get_mpp(2.0) == 0.25


def test_get_scaling(sample_image_reader: TestFileImageReader) -> None:
    sample_image_reader._open_file()
    assert sample_image_reader.get_scaling(None) == 1.0
    assert sample_image_reader.get_scaling(0.25) == 2.0


def test_stitching_modes(temp_h5_file: Path) -> None:
    stitching_modes = [StitchingMode.AVERAGE, StitchingMode.MAXIMUM]

    for stitching_mode in stitching_modes:
        reader = TestFileImageReader(temp_h5_file, stitching_mode)
        reader._open_file()

        with patch("pyvips.Image.new_from_array") as mock_pyvips:
            # Reading a region that covers all 4 tiles
            _ = reader.read_region((0, 0), 0, (350, 350))
            mock_pyvips.assert_called_once()

            stitched_image = mock_pyvips.call_args[0][0]

            assert stitched_image.shape == (350, 350, 3)  # Shape should match the requested region

            # Convert stitched_image from (H, W, C) to (C, H, W) for easier color checks
            stitched_image = stitched_image.transpose(2, 0, 1)

            # Check the non-overlapping regions
            red_colored, green_colored, blue_colored, yellow_colored = create_colored_tiles((200, 200))

            expected_red, expected_green, expected_blue, expected_yellow = (
                red_colored[:150, :150, :],
                green_colored[:150, :150, :],
                blue_colored[:150, :150, :],
                yellow_colored[:150, :150, :],
            )

            assert np.all(stitched_image[:, :150, :150] == expected_red.transpose(2, 0, 1))  # Top-left tile (Red)
            assert np.all(stitched_image[:, :150, 200:] == expected_green.transpose(2, 0, 1))  # Top-right tile (Green)
            assert np.all(stitched_image[:, 200:, :150] == expected_blue.transpose(2, 0, 1))  # Bottom-left tile (Blue)
            assert np.all(
                stitched_image[:, 200:, 200:] == expected_yellow.transpose(2, 0, 1)
            )  # Bottom-right tile (Yellow)

            if stitching_mode == StitchingMode.AVERAGE:
                # Check the overlapping regions for average value
                overlap_color_top = (
                    red_colored.transpose(2, 0, 1)[:, :150, 150:200]
                    + green_colored.transpose(2, 0, 1)[:, :150, 150:200]
                ) / 2
                overlap_color_left = (
                    red_colored.transpose(2, 0, 1)[:, 150:200, :150] + blue_colored.transpose(2, 0, 1)[:, 150:200, :150]
                ) / 2
                overlap_color_bottom = (
                    blue_colored.transpose(2, 0, 1)[:, :150, 150:200]
                    + yellow_colored.transpose(2, 0, 1)[:, :150, 150:200]
                ) / 2
                overlap_color_right = (
                    green_colored.transpose(2, 0, 1)[:, 150:200, :150]
                    + yellow_colored.transpose(2, 0, 1)[:, 150:200, :150]
                ) / 2
                overlap_color_center = (
                    red_colored.transpose(2, 0, 1)[:, 150:200, 150:200]
                    + green_colored.transpose(2, 0, 1)[:, 150:200, 150:200]
                    + blue_colored.transpose(2, 0, 1)[:, 150:200, 150:200]
                    + yellow_colored.transpose(2, 0, 1)[:, 150:200, 150:200]
                ) / 4

            else:
                # Check the overlapping regions for maximum value
                overlap_color_top = np.maximum(
                    red_colored.transpose(2, 0, 1)[:, :150, 150:200], green_colored.transpose(2, 0, 1)[:, :150, 150:200]
                )
                overlap_color_left = np.maximum(
                    red_colored.transpose(2, 0, 1)[:, 150:200, :150], blue_colored.transpose(2, 0, 1)[:, 150:200, :150]
                )
                overlap_color_bottom = np.maximum(
                    blue_colored.transpose(2, 0, 1)[:, :150, 150:200],
                    yellow_colored.transpose(2, 0, 1)[:, :150, 150:200],
                )
                overlap_color_right = np.maximum(
                    green_colored.transpose(2, 0, 1)[:, 150:200, :150],
                    yellow_colored.transpose(2, 0, 1)[:, 150:200, :150],
                )
                overlap_color_center = np.maximum.reduce(
                    [
                        red_colored.transpose(2, 0, 1)[:, 150:200, 150:200],
                        green_colored.transpose(2, 0, 1)[:, 150:200, 150:200],
                        blue_colored.transpose(2, 0, 1)[:, 150:200, 150:200],
                        yellow_colored.transpose(2, 0, 1)[:, 150:200, 150:200],
                    ]
                )

            assert np.all(stitched_image[:, :150, 150:200] == overlap_color_top)  # Top overlap
            assert np.all(stitched_image[:, 150:200, :150] == overlap_color_left)  # Left overlap
            assert np.all(stitched_image[:, 200:350, 150:200] == overlap_color_bottom)  # Bottom overlap
            assert np.all(stitched_image[:, 150:200, 200:350] == overlap_color_right)  # Right overlap
            assert np.all(stitched_image[:, 150:200, 150:200] == overlap_color_center)  # Center overlap


def test_context_manager(sample_image_reader: TestFileImageReader) -> None:
    with patch.object(sample_image_reader, "_open_file", wraps=sample_image_reader._open_file) as mock_open:
        with sample_image_reader as reader:
            mock_open.assert_called_once()
        assert reader._file is not None


def test_open_file_raises(sample_image_reader: TestFileImageReader) -> None:
    with patch.object(Path, "is_file", return_value=False):
        with pytest.raises(FileNotFoundError):
            sample_image_reader._open_file()


def test_empty_tile(sample_image_reader: TestFileImageReader) -> None:
    sample_image_reader._open_file()
    empty_tile = sample_image_reader._empty_tile()
    assert empty_tile.shape == (3, 200, 200)
