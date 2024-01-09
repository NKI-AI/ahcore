"""
Reader classes.

- `H5FileImageReader`: to read files written using the `ahcore.writers.H5FileImageWriter`.

"""
import errno
import json
import math
import os
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional, Type, cast

import h5py
import numpy as np
from scipy.ndimage import map_coordinates

from ahcore.utils.io import get_logger
from ahcore.utils.types import BoundingBoxType, GenericArray

logger = get_logger(__name__)


class StitchingMode(str, Enum):
    CROP = "crop"
    AVERAGE = "average"
    MAXIMUM = "maximum"


def crop_to_bbox(array: GenericArray, bbox: BoundingBoxType) -> GenericArray:
    (start_x, start_y), (width, height) = bbox
    return array[:, start_y : start_y + height, start_x : start_x + width]


class H5FileImageReader:
    def __init__(self, filename: Path, stitching_mode: StitchingMode) -> None:
        self._filename = filename
        self._stitching_mode = stitching_mode

        self.__empty_tile: GenericArray | None = None

        self._h5file: Optional[h5py.File] = None
        self._metadata = None
        self._mpp = None
        self._tile_size = None
        self._tile_overlap = None
        self._size = None
        self._num_channels = None
        self._dtype = None
        self._stride = None

    @classmethod
    def from_file_path(cls, filename: Path, stitching_mode: StitchingMode = StitchingMode.CROP) -> "H5FileImageReader":
        return cls(filename=filename, stitching_mode=stitching_mode)

    @property
    def size(self) -> tuple[int, int]:
        if not self._size:
            self._open_file()
            assert self._size
        return self._size

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

    def _open_file(self) -> None:
        if not self._filename.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(self._filename))

        try:
            self._h5file = h5py.File(self._filename, "r")
        except OSError as e:
            logger.error(f"Could not open file {self._filename}: {e}")
            raise e

        try:
            self._metadata = json.loads(self._h5file.attrs["metadata"])
        except KeyError as e:
            logger.error(f"Could not read metadata from file {self._filename}: {e}")
            raise e

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
        self._stride = (
            self._tile_size[0] - self._tile_overlap[0],
            self._tile_size[1] - self._tile_overlap[1],
        )

        if self._metadata["has_color_profile"]:
            _color_profile = self._h5file["color_profile"][()].tobytes()
            raise NotImplementedError(f"Color profiles are not yet implemented, and are present in {self._filename}.")

    def __enter__(self) -> "H5FileImageReader":
        if self._h5file is None:
            self._open_file()
        return self

    def _empty_tile(self) -> GenericArray:
        if self.__empty_tile is not None:
            return self.__empty_tile

        # When this happens we would already be in the read_region, and self._num_channels would be populated.
        assert self._num_channels

        self.__empty_tile = np.zeros((self._num_channels, *self._tile_size), dtype=self._dtype)
        return self.__empty_tile

    def read_region(
        self,
        location: tuple[int, int],
        scaling: float,
        size: tuple[int, int],
    ) -> GenericArray:
        """

        Parameters
        ----------
        location : tuple[int, int]
            Location from the top left (x, y) in pixel coordinates given at the requested scaling.
        scaling : float
        size : tuple[int, int]
            Size of the output region

        Returns
        -------
        np.ndarray
            The requested region.
        """
        if scaling == 1.0:
            return self.read_region_raw(location, size)

        order = 1
        # Calculate original location and size considering the scaling

        # unpack for mypy
        l1, l2 = location
        s1, s2 = size

        original_location = (
            int(math.floor(l1 / scaling)) - order,
            int(math.floor(l2 / scaling)) - order,
        )
        original_size = (
            int(math.ceil(s1 / scaling)) + order,
            int(math.ceil(s2 / scaling)) + order,
        )

        raw_region = self.read_region_raw(original_location, original_size)

        # Determine the fractional start and end coordinates for mapping
        fractional_start = tuple(map(lambda _, ol: (_ / scaling) - ol + order, location, original_location))
        fractional_end = tuple(fs + size[i] / scaling for i, fs in enumerate(fractional_start))

        # Create an array of coordinates for map_coordinates
        # mypy doesn't properly understand yet that the complex type is valid
        coordinates = np.mgrid[
            fractional_start[0] : fractional_end[0] : complex(size[0]),  # type: ignore
            fractional_start[1] : fractional_end[1] : complex(size[1]),  # type: ignore
        ]
        coordinates = np.moveaxis(coordinates, 0, -1)

        # Interpolate using map_coordinates for all channels
        grid = np.mgrid[: raw_region.shape[0]]
        coordinates = np.concatenate([grid[:, None, None], coordinates], axis=0)
        # scipy doesn't have proper typing yet
        rescaled_region = cast(GenericArray, map_coordinates(raw_region, coordinates, order=order))

        return rescaled_region

    def read_region_raw(self, location: tuple[int, int], size: tuple[int, int]) -> GenericArray:
        """
        Reads a region in the stored h5 file. This function stitches the regions as saved in the h5 file. Doing this
        it takes into account:
        1) The region overlap, several region merging strategies are implemented: cropping, averaging across borders
          and taking the maximum across borders.
        2) If tiles are saved or not. In case the tiles are skipped due to a background mask, an empty tile is returned.

        Parameters
        ----------
        location : tuple[int, int]
            Coordinates (x, y) of the upper left corner of the region.
        size : tuple[int, int]
            The (h, w) size of the extracted region.

        Returns
        -------
        np.ndarray
            Extracted region
        """
        if self._h5file is None:
            self._open_file()
        assert self._h5file, "File is not open. Should not happen"
        assert self._tile_size
        assert self._tile_overlap

        image_dataset = self._h5file["data"]
        num_tiles = self._metadata["num_tiles"]
        tile_indices = self._h5file["tile_indices"]

        total_rows = math.ceil((self._size[1] - self._tile_overlap[1]) / self._stride[1])
        total_cols = math.ceil((self._size[0] - self._tile_overlap[0]) / self._stride[0])

        assert total_rows * total_cols == num_tiles

        x, y = location
        w, h = size
        if x < 0 or y < 0 or x + w > self._size[0] or y + h > self._size[1]:
            logger.error(f"Requested region is out of bounds: {location}, {self._size}")
            raise ValueError("Requested region is out of bounds")

        start_row = y // self._stride[1]
        end_row = min((y + h - 1) // self._stride[1] + 1, total_rows)
        start_col = x // self._stride[0]
        end_col = min((x + w - 1) // self._stride[0] + 1, total_cols)

        if self._stitching_mode == StitchingMode.AVERAGE:
            divisor_array = np.zeros((h, w), dtype=np.uint8)
        stitched_image = np.zeros((self._num_channels, h, w), dtype=self._dtype)
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                tile_idx = (i * total_cols) + j
                # Map through tile indices
                tile_index_in_image_dataset = tile_indices[tile_idx]
                tile = (
                    self._empty_tile()
                    if tile_index_in_image_dataset == -1
                    else image_dataset[tile_index_in_image_dataset]
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
                    raise NotImplementedError
                    tile_start_y = max(0, -start_y)
                    tile_end_y = img_end_y - img_start_y
                    tile_start_x = max(0, -start_x)
                    tile_end_x = img_end_x - img_start_x

                    # TODO: Replace this with crop_to_bbox
                    cropped_tile = tile[tile_start_y:tile_end_y, tile_start_x:tile_end_x]
                    stitched_image[img_start_y:img_end_y, img_start_x:img_end_x] += cropped_tile
                    divisor_array[img_start_y:img_end_y, img_start_x:img_end_x] += 1
                else:
                    raise ValueError("Unsupported stitching mode")

        if self._stitching_mode == StitchingMode.AVERAGE:
            stitched_image = (stitched_image / divisor_array[..., np.newaxis]).astype(float)

        if self._precision != "FP32":
            # Always convert to float32.
            stitched_image = stitched_image / self._multiplier
            stitched_image = stitched_image.astype(np.float32)

        return stitched_image

    def close(self) -> None:
        if self._h5file is not None:
            self._h5file.close()  # Close the file in close
        del self._h5file  # Reset the h5file attribute

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        self.close()
        return False
