from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Type

import numpy as np
from dlup.writers import TiffCompression, TifffileImageWriter
from numpy import typing as npt

from ahcore.callbacks.converters.common import ConvertCallbacks
from ahcore.readers import FileImageReader, StitchingMode
from ahcore.utils.callbacks import _ValidationDataset
from ahcore.utils.io import get_logger
from ahcore.utils.types import GenericNumberArray

logger = get_logger(__name__)


def _tile_process_function(x: GenericNumberArray) -> npt.NDArray[np.int_]:
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
    def __init__(
        self,
        reader_class: Type[FileImageReader],
        colormap: dict[int, str],
        max_concurrent_tasks: int = 1,
        tile_size: tuple[int, int] = (1024, 1024),
    ) -> None:
        """
        Write tiffs based on the outputs stored by FileImageWriter.

        Parameters
        ----------
        reader_class : Type[FileImageReader]
            The reader class to use to read the images, e.g., H5FileImageReader or ZarrFileImageReader.
        colormap : dict[int, str]
            A dictionary mapping class indices to RGB colors. This will be processed by libtiff as coloring the output
            of the segmentation map.
        max_concurrent_tasks : int
            The maximum number of concurrent processes to write the tiff files.
        tile_size : tuple[int, int]
            The size of the tiles to write the tiff files in.

        Returns
        -------
        None

        """
        self._reader_class = reader_class
        self._colormap = colormap
        self._tile_size = tile_size
        self._tile_process_function = _tile_process_function
        self.has_returns = False
        super().__init__(max_concurrent_tasks=max_concurrent_tasks)

    def process_task(self, filename: Path, cache_filename: Path) -> None:
        assert self._tile_process_function == _tile_process_function
        _write_tiff(
            cache_filename,
            self._tile_size,
            self._colormap,
            self._reader_class,
            _tiff_iterator_from_reader,
        )


def _iterator_from_reader(
    cache_reader: FileImageReader,
    tile_size: tuple[int, int],
    tile_process_function: Callable[[GenericNumberArray], GenericNumberArray] | None,
) -> Iterator[GenericNumberArray]:
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


def _tiff_iterator_from_reader(
    cache_reader: FileImageReader,
    tile_size: tuple[int, int],
) -> Iterator[npt.NDArray[np.int_]]:
    iterator = _iterator_from_reader(cache_reader, tile_size, _tile_process_function)

    for sample in iterator:
        yield sample.astype(np.uint8)


def _write_tiff(
    filename: Path,
    tile_size: tuple[int, int],
    colormap: dict[int, str] | None,
    file_reader: Type[FileImageReader],
    iterator_from_reader: Callable[[FileImageReader, tuple[int, int]], Iterator[npt.NDArray[np.int_]]],
) -> None:
    with file_reader(filename, stitching_mode=StitchingMode.AVERAGE, tile_filter=(5, 5)) as cache_reader:
        writer = TifffileImageWriter(
            filename.with_suffix(".tiff"),
            size=cache_reader.size,
            mpp=cache_reader.mpp,
            tile_size=tile_size,
            pyramid=True,
            compression=TiffCompression.ZSTD,
            quality=100,
            colormap=colormap,
        )
        writer.from_tiles_iterator(iterator_from_reader(cache_reader, tile_size))
