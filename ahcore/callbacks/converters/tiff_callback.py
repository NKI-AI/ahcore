from __future__ import annotations

from pathlib import Path
from typing import Callable, Generator, Iterator

import numpy as np
from dlup._image import Resampling
from dlup.writers import TiffCompression, TifffileImageWriter
from numpy import typing as npt

from ahcore.callbacks.converters.common import ConvertCallbacks
from ahcore.readers import FileImageReader, StitchingMode
from ahcore.utils.callbacks import _ValidationDataset
from ahcore.utils.io import get_logger
from ahcore.utils.types import GenericNumberArray

logger = get_logger(__name__)


def _tile_process_function(x: GenericNumberArray) -> GenericNumberArray:
    """
    Function to process a tile before writing it to a tiff file.

    Arguments
    ---------
    x : GenericNumberArray
        A 2D array representing an output tile (C, H, W).

    Returns
    -------
    GenericNumberArray
        A 2D array argmaxed along the first axis and casted to uint8, with dimensions (H, W).
    """
    return np.asarray(np.argmax(x, axis=0).astype(np.uint8))


class TiffConverterCallback(ConvertCallbacks):
    def __init__(self, reader_class, colormap, max_concurrent_tasks: int = 1):
        self._reader_class = reader_class
        self._colormap = colormap
        self._tile_size = (1024, 1024)
        self._tile_process_function = _tile_process_function
        super().__init__(max_concurrent_tasks=max_concurrent_tasks)

    def process_task(self, filename: Path, cache_filename: Path) -> None:
        _write_tiff(
            cache_filename,
            self._tile_size,
            self._tile_process_function,
            self._colormap,
            self._reader_class,
            _generator_from_reader,
        )


def _generator_from_reader(
    cache_reader: FileImageReader,
    tile_size: tuple[int, int],
    tile_process_function: Callable[[GenericNumberArray], GenericNumberArray],
) -> Generator[GenericNumberArray, None, None]:
    validation_dataset = _ValidationDataset(
        data_description=None,
        native_mpp=cache_reader.mpp,
        reader=cache_reader,
        annotations=None,
        mask=None,
        region_size=tile_size,
    )

    for sample in validation_dataset:
        region = sample["prediction"]
        yield region if tile_process_function is None else tile_process_function(region)


def _write_tiff(
    filename: Path,
    tile_size: tuple[int, int],
    tile_process_function: Callable[[GenericNumberArray], GenericNumberArray],
    colormap: dict[int, str] | None,
    file_reader: FileImageReader,
    generator_from_reader: Callable[
        [FileImageReader, tuple[int, int], Callable[[GenericNumberArray], GenericNumberArray]],
        Iterator[npt.NDArray[np.int_]],
    ],
) -> None:
    with file_reader(filename, stitching_mode=StitchingMode.CROP) as cache_reader:
        writer = TifffileImageWriter(
            filename.with_suffix(".tiff"),
            size=cache_reader.size,
            mpp=cache_reader.mpp,
            tile_size=tile_size,
            pyramid=True,
            compression=TiffCompression.ZSTD,
            quality=100,
            interpolator=Resampling.NEAREST,
            colormap=colormap,
        )
        writer.from_tiles_iterator(generator_from_reader(cache_reader, tile_size, tile_process_function))
