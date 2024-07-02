"""Ahcore's callbacks"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import numpy.typing as npt
from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.data.transforms import convert_annotations, rename_labels
from dlup.tiling import Grid, GridOrder, TilingMode
from shapely.geometry import MultiPoint, Point
from torch.utils.data import Dataset

from ahcore.readers import FileImageReader
from ahcore.transforms.pre_transforms import one_hot_encoding
from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger
from ahcore.utils.types import DlupDatasetSample

logger = get_logger(__name__)

logging.getLogger("pyvips").setLevel(logging.ERROR)


def _validate_annotations(data_description, annotations: Optional[WsiAnnotations]) -> Optional[WsiAnnotations]:
    if annotations is None:
        return None

    if isinstance(annotations, WsiAnnotations):
        if data_description is None:
            raise ValueError(
                "Annotations as a `WsiAnnotations` class are provided but no data description is given."
                "This is required to map the labels to indices."
            )
    elif isinstance(annotations, SlideImage):
        pass  # We do not need a specific test for this
    else:
        raise NotImplementedError(f"Annotations of type {type(annotations)} are not supported.")

    return annotations


class _ValidationDataset(Dataset[DlupDatasetSample]):
    """Helper dataset to compute the validation metrics."""

    def __init__(
        self,
        data_description: Optional[DataDescription],
        native_mpp: float,
        reader: FileImageReader,
        annotations: Optional[WsiAnnotations] = None,
        mask: Optional[WsiAnnotations] = None,
        region_size: tuple[int, int] = (1024, 1024),
    ):
        """
        Parameters
        ----------
        data_description : DataDescription
        native_mpp : float
            The actual mpp of the underlying image.
        reader : H5FileImageReader
        annotations : WsiAnnotations
        mask : WsiAnnotations
        region_size : tuple[int, int]
            The region size to use to split up the image into regions.
        """
        super().__init__()
        self._data_description = data_description
        self._native_mpp = native_mpp
        self._scaling = self._native_mpp / reader.mpp
        self._reader = reader
        self._region_size = region_size
        self._logger = get_logger(type(self).__name__)

        self._annotations = _validate_annotations(data_description, annotations)
        self._mask = _validate_annotations(data_description, mask)

        self._grid = Grid.from_tiling(
            (0, 0),
            reader.size,
            tile_size=self._region_size,
            tile_overlap=(0, 0),
            mode=TilingMode.overflow,
            order=GridOrder.C,
        )

        self._regions = self._generate_regions()
        self._logger.debug(f"Number of validation regions: {len(self._regions)}")

    def _generate_regions(self) -> list[tuple[int, int]]:
        """Generate the regions to use. These regions are filtered grid cells where there is a mask.

        Returns
        -------
        list[tuple[int, int]]
            The list of regions.
        """
        regions = []
        for coordinates in self._grid:
            _coordinates = (coordinates[0], coordinates[1])
            if self._mask is None or self._is_masked(_coordinates):
                regions.append(_coordinates)
        return regions

    def _is_masked(self, coordinates: tuple[int, int]) -> bool:
        """Check if the region is masked. This works with any masking function that supports a `read_region` method or
        returns a list of annotations with an `area` attribute. In case there are elements of the form `Point` in the
        annotation list, these are also added.

        Parameters
        ----------
        coordinates : tuple[int, int]
            The coordinates of the region to check.

        Returns
        -------
        bool
            True if the region is masked, False otherwise. Will also return True when there is no mask.
        """
        if self._mask is None:
            return True

        region_mask = self._mask.read_region(coordinates, self._scaling, self._region_size)

        if isinstance(region_mask, np.ndarray):
            return region_mask.sum() > 0

        # We check if the region is not a Point, otherwise this annotation is always included
        # Else, we compute if there is a positive area in the region.
        return bool(sum(_.area if _ is not isinstance(_, (Point, MultiPoint)) else 1.0 for _ in region_mask) > 0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = {}
        coordinates = self._regions[idx]

        sample["prediction"] = self._get_h5_region(coordinates)

        if self._annotations is not None:
            target, roi = self._get_annotation_data(coordinates)
            if roi is not None:
                sample["roi"] = roi.astype(np.uint8)
            else:
                sample["roi"] = None  # type: ignore
            sample["target"] = target

        return sample

    def _get_h5_region(self, coordinates: tuple[int, int]) -> npt.NDArray[np.uint8 | np.uint16 | np.float32 | np.bool_]:
        x, y = coordinates
        width, height = self._region_size

        if x + width > self._reader.size[0] or y + height > self._reader.size[1]:
            region = self._read_and_pad_region(coordinates)
        else:
            region = self._reader.read_region(coordinates, 0, self._region_size).numpy().transpose((2, 0, 1))
        return region

    def _read_and_pad_region(self, coordinates: tuple[int, int]) -> npt.NDArray[Any]:
        x, y = coordinates
        width, height = self._region_size
        new_width = min(width, self._reader.size[0] - x)
        new_height = min(height, self._reader.size[1] - y)
        clipped_region = self._reader.read_region((x, y), 0, (new_width, new_height))

        prediction = np.zeros((clipped_region.shape[0], *self._region_size), dtype=clipped_region.dtype)
        prediction[:, :new_height, :new_width] = clipped_region
        return prediction

    def _get_annotation_data(
        self, coordinates: tuple[int, int]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int_] | None]:
        if not self._annotations:
            raise ValueError("No annotations are provided.")

        if not self._data_description:
            raise ValueError("No data description is provided.")

        if not self._data_description.index_map:
            raise ValueError("Index map is not provided.")

        _annotations = self._annotations.read_region(coordinates, self._scaling, self._region_size)

        if self._data_description.remap_labels:
            _annotations = rename_labels(_annotations, remap_labels=self._data_description.remap_labels)

        points, region, roi = convert_annotations(
            _annotations,
            self._region_size,
            index_map=self._data_description.index_map,
            roi_name=self._data_description.roi_name,
        )
        encoded_region = one_hot_encoding(index_map=self._data_description.index_map, mask=region)
        if roi is not None:
            return encoded_region, roi[np.newaxis, ...]
        return encoded_region, None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self) -> int:
        return len(self._regions)


def _get_uuid_for_filename(input_path: Path) -> str:
    """Get a unique filename for the given input path. This is done by hashing the absolute path of the file.
    This is required because we cannot assume any input format. We hash the complete input path.

    Parameters
    ----------
    input_path : Path
        The input path to hash.

    Returns
    -------
    str
        The hashed filename.
    """
    # Get the absolute path of the file
    input_path = Path(input_path).resolve()

    # Create a SHA256 hash of the file path
    hash_object = hashlib.sha256(str(input_path).encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def get_output_filename(dump_dir: Path, input_path: Path, model_name: str, counter: str) -> Path:
    hex_dig = _get_uuid_for_filename(input_path=input_path)

    # Return the hashed filename with the new extension
    if counter is not None:
        return dump_dir / "outputs" / model_name / f"{counter}" / f"{hex_dig}.cache"
    return dump_dir / "outputs" / model_name / f"{hex_dig}.cache"
