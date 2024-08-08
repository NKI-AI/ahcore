"""
Reader classes.

- `H5FileImageReader`: to read files written using the `ahcore.writers.H5FileImageWriter`.

"""

import abc
import errno
import io
import json
import math
import os
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Optional, Type

import h5py
import numpy as np
import PIL
import pyvips
import zarr
from zarr.storage import ZipStore

from ahcore.utils.io import get_logger
from ahcore.utils.types import BoundingBoxType, GenericNumberArray, InferencePrecision

logger = get_logger(__name__)


class StitchingMode(str, Enum):
    CROP = "CROP"
    AVERAGE = "AVERAGE"
    MAXIMUM = "MAXIMUM"


def crop_to_bbox(array: GenericNumberArray, bbox: BoundingBoxType) -> GenericNumberArray:
    (start_x, start_y), (width, height) = bbox
    return array[:, start_y : start_y + height, start_x : start_x + width]


class FileImageReader(abc.ABC):
    def __init__(self, filename: Path, stitching_mode: StitchingMode) -> None:
        self._filename = filename
        self._stitching_mode = stitching_mode
        self.__empty_tile: GenericNumberArray | None = None

        self._file: Optional[Any] = None
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

    @classmethod
    def from_file_path(cls, filename: Path, stitching_mode: StitchingMode = StitchingMode.CROP) -> "FileImageReader":
        return cls(filename=filename, stitching_mode=stitching_mode)

    @property
    def size(self) -> tuple[int, int]:
        if not self._size:
            self._open_file()
            assert self._size
        return tuple(self._size)

    @property
    def mpp(self) -> float:
        if not self._mpp:
            self._open_file()
            assert self._mpp
        return self._mpp

    def get_mpp(self, scaling: Optional[float]) -> float:
        if not self._mpp:
            self._open_file()
            assert self._mpp
        if scaling is None:
            return self.mpp

        return self._mpp / scaling

    def get_scaling(self, mpp: Optional[float]) -> float:
        """Inverse of get_mpp()."""
        if not self._mpp:
            self._open_file()
            assert self._mpp
        if not mpp:
            return 1.0
        return self._mpp / mpp

    @abc.abstractmethod
    def _open_file_handle(self, filename: Path) -> Any:
        pass

    @abc.abstractmethod
    def _read_metadata(self) -> None:
        pass

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

        if self._metadata["has_color_profile"]:
            _color_profile = self._file["color_profile"][()].tobytes()
            logger.warning(
                f"Color profiles are not yet implemented, but {_color_profile} is present in {self._filename}."
            )

    def __enter__(self) -> "FileImageReader":
        if self._file is None:
            self._open_file()
        return self

    def _empty_tile(self) -> GenericNumberArray:
        if self.__empty_tile is not None:
            return self.__empty_tile

        # When this happens we would already be in the read_region, and self._num_channels would be populated.
        assert self._num_channels

        self.__empty_tile = np.zeros((self._num_channels, *self._tile_size), dtype=self._dtype)
        return self.__empty_tile

    @property
    def metadata(self) -> dict[str, Any]:
        if not self._metadata:
            self._open_file()
            assert self._metadata
        return self._metadata

    def _decompress_data(self, tile: GenericNumberArray) -> GenericNumberArray:
        if self._is_binary:
            with PIL.Image.open(io.BytesIO(tile)) as img:
                return np.array(img).transpose(2, 0, 1)
        else:
            return tile

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> pyvips.Image:
        """
        Reads a region in the stored h5 file. This function stitches the regions as saved in the cache file. Doing this
        it takes into account:
        1) The region overlap, several region merging strategies are implemented: cropping, averaging across borders
          and taking the maximum across borders.
        2) If tiles are saved or not. In case the tiles are skipped due to a background mask, an empty tile is returned.

        Parameters
        ----------
        location : tuple[int, int]
            Coordinates (x, y) of the upper left corner of the region.
        level : int
            The level of the region. Only level 0 is supported.
        size : tuple[int, int]
            The (h, w) size of the extracted region.

        Returns
        -------
        pyvips.Image
            Extracted region
        """
        if level != 0:
            raise ValueError("Only level 0 is supported")

        if self._file is None:
            self._open_file()
        assert self._file, "File is not open. Should not happen"
        assert self._tile_size is not None, "self._tile_size should not be None"
        assert self._tile_overlap is not None, "self._tile_overlap should not be None"

        image_dataset = self._file["data"]
        num_tiles = self._metadata["num_tiles"]
        tile_indices = self._file["tile_indices"]

        total_rows = math.ceil((self._size[1] - self._tile_overlap[1]) / self._stride[1])
        total_cols = math.ceil((self._size[0] - self._tile_overlap[0]) / self._stride[0])

        assert total_rows * total_cols == num_tiles

        x, y = location
        w, h = size
        if x < 0 or y < 0 or x + w > self._size[0] or y + h > self._size[1]:
            logger.error(f"Requested region is out of bounds: {location}, {self._size}")
            raise ValueError("Requested region is out of bounds")

        if self._stitching_mode not in StitchingMode:
            raise ValueError(f"Stitching mode {self._stitching_mode} is not supported.")

        start_row = y // self._stride[1]
        end_row = min((y + h - 1) // self._stride[1] + 1, total_rows)
        start_col = x // self._stride[0]
        end_col = min((x + w - 1) // self._stride[0] + 1, total_cols)

        stitched_image = np.zeros((self._num_channels, h, w), dtype=self._dtype)

        if self._stitching_mode == StitchingMode.AVERAGE:
            average_mask = np.zeros((h, w), dtype=self._dtype)

        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                tile_idx = (i * total_cols) + j
                # Map through tile indices
                tile_index_in_image_dataset = tile_indices[tile_idx]
                tile = (
                    self._empty_tile()
                    if tile_index_in_image_dataset == -1
                    else self._decompress_data(image_dataset[tile_index_in_image_dataset])
                )
                start_y = i * self._stride[1] - y
                end_y = start_y + self._tile_size[1]
                start_x = j * self._stride[0] - x
                end_x = start_x + self._tile_size[0]

                img_start_y = max(0, start_y)
                img_end_y = min(h, end_y)
                img_start_x = max(0, start_x)
                img_end_x = min(w, end_x)

                if self._stitching_mode == StitchingMode.CROP:
                    crop_start_y = img_start_y - start_y
                    crop_end_y = img_end_y - start_y
                    crop_start_x = img_start_x - start_x
                    crop_end_x = img_end_x - start_x

                    bbox = (crop_start_x, crop_start_y), (
                        crop_end_x - crop_start_x,
                        crop_end_y - crop_start_y,
                    )
                    cropped_tile = crop_to_bbox(tile, bbox)
                    stitched_image[:, img_start_y:img_end_y, img_start_x:img_end_x] = cropped_tile

                elif self._stitching_mode == StitchingMode.AVERAGE:
                    tile_start_y = max(0, -start_y)
                    tile_end_y = min(self._tile_size[1], h - start_y)
                    tile_start_x = max(0, -start_x)
                    tile_end_x = min(self._tile_size[0], w - start_x)

                    average_mask[img_start_y:img_end_y, img_start_x:img_end_x] += 1
                    stitched_image[:, img_start_y:img_end_y, img_start_x:img_end_x] += tile[
                        :, tile_start_y:tile_end_y, tile_start_x:tile_end_x
                    ]

                elif self._stitching_mode == StitchingMode.MAXIMUM:
                    tile_start_y = max(0, -start_y)
                    tile_end_y = min(self._tile_size[1], h - start_y)
                    tile_start_x = max(0, -start_x)
                    tile_end_x = min(self._tile_size[0], w - start_x)

                    if i == start_row and j == start_col:
                        # The first tile cannot be compared with anything. So, we just copy it.
                        stitched_image[:, img_start_y:img_end_y, img_start_x:img_end_x] = tile[
                            :, tile_start_y:tile_end_y, tile_start_x:tile_end_x
                        ]

                    stitched_image[:, img_start_y:img_end_y, img_start_x:img_end_x] = np.maximum(
                        stitched_image[:, img_start_y:img_end_y, img_start_x:img_end_x],
                        tile[:, tile_start_y:tile_end_y, tile_start_x:tile_end_x],
                    )

        # Adjust the precision and convert to float32 before averaging to avoid loss of precision.
        if self._precision != str(InferencePrecision.UINT8) or self._stitching_mode == StitchingMode.AVERAGE:
            stitched_image = stitched_image / self._multiplier
            stitched_image = stitched_image.astype(np.float32)

        if self._stitching_mode == StitchingMode.AVERAGE:
            overlap_regions = average_mask > 0
            # Perform division to average the accumulated pixel values
            stitched_image[:, overlap_regions] = stitched_image[:, overlap_regions] / average_mask[overlap_regions]

        return pyvips.Image.new_from_array(stitched_image.transpose(1, 2, 0))

    @abc.abstractmethod
    def close(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        self.close()
        return False


class H5FileImageReader(FileImageReader):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._file: Optional[h5py.File] = None

    def _open_file_handle(self, filename: Path) -> h5py.File:
        try:
            file = h5py.File(self._filename, "r")
        except OSError as e:
            logger.error(f"Could not open file {self._filename}: {e}")
            raise e
        return file

    def _read_metadata(self) -> None:
        if not self._file:
            raise ValueError("File is not open.")

        try:
            self._metadata = json.loads(self._file.attrs["metadata"])
        except KeyError as e:
            logger.error(f"Could not read metadata from file {self._filename}: {e}")
            raise e

    def close(self) -> None:
        if self._file is not None:
            self._file.close()  # Close the file in close
        del self._file  # Reset the file attribute


class ZarrFileImageReader(FileImageReader):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._file: Optional[zarr.Group] = None

    def _open_file_handle(self, filename: Path) -> zarr.Group:
        try:
            zip_store = ZipStore(str(self._filename), mode="r")
            file = zarr.open_group(store=zip_store, mode="r")
        except OSError as e:
            logger.error(f"Could not open file {self._filename}: {e}")
            raise e
        return file

    def _read_metadata(self) -> None:
        if not self._file:
            raise ValueError("File is not open.")
        try:
            self._metadata = self._file.attrs.asdict()
        except KeyError as e:
            logger.error(f"Could not read metadata from file {self._filename}: {e}")
            raise e

    def close(self) -> None:
        del self._file  # Reset the file attribute
