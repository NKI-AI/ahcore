from __future__ import annotations

import abc
import pathlib
from pathlib import Path
from typing import NamedTuple, Any, Callable, Generator, Iterator
from threading import Semaphore

import numpy as np
import pytorch_lightning as pl
from dlup._image import Resampling
from dlup.writers import TifffileImageWriter, TiffCompression
from numpy import typing as npt
from pytorch_lightning import Callback

from ahcore.readers import FileImageReader, StitchingMode
from ahcore.utils.io import get_logger
from ahcore.utils.callbacks import get_output_filename, _ValidationDataset
from ahcore.utils.types import GenericNumberArray

logger = get_logger(__name__)

class ConvertCallbacks(abc.ABC):
    def __init__(self, max_concurrent_tasks: int=1):
        self._max_concurrent_tasks = max_concurrent_tasks
        self._semaphore = Semaphore(max_concurrent_tasks)
        self._callback = None
        self._trainer = None
        self._pl_module = None
        self._stage = None
        self._dump_dir = None

        self._storage = {}

    def setup(self, callback: Callback, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self._callback = callback
        self._trainer = trainer
        self._pl_module = pl_module
        self._stage = stage

        self._dump_dir = self._callback.dump_dir

    @property
    def dump_dir(self) -> pathlib.Path:
        return self._dump_dir

    @abc.abstractmethod
    def schedule_task(self, filename: pathlib.Path, cache_filename: pathlib.Path) -> None:
        """Abstract method to schedule the task"""

    def start(self, filename: str) -> None:
        cache_filename = get_output_filename(
            dump_dir=self._callback.dump_dir,
            input_path=self._pl_module.data_description.data_dir / filename,
            model_name=str(self._pl_module.name),
            step=self._pl_module.global_step,
        )

        # This should never not exist.
        assert pathlib.Path(filename).exists()
        assert cache_filename.exists()

        self.schedule_task(filename=pathlib.Path(filename), cache_filename=cache_filename)


class TiffConverterCallback(ConvertCallbacks):
    def __init__(self, reader_class, colormap, max_concurrent_tasks: int=1):
        self._reader_class = reader_class
        self._colormap = colormap
        self._tile_size = (1024, 1024)
        self._tile_process_function = None
        super().__init__(max_concurrent_tasks=max_concurrent_tasks)

    def schedule_task(self, filename: pathlib.Path, cache_filename: pathlib.Path) -> None:
        logger.info("Scheduling task for %s, using cache %s and writing to %s", filename, cache_filename, output_filename)

        self._semaphore.acquire()
        # Start the task
        _write_tiff(
            cache_filename,
            self._tile_size,
            self._tile_process_function,
            self._colormap,
            self._reader_class,
            _generator_from_reader,
        )
        self._semaphore.release()


class CallbackOutput(NamedTuple):
    metrics: Any

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
        region_size=(1024, 1024),
    )

    for sample in validation_dataset:
        region = sample["prediction"]
        yield region if tile_process_function is None else tile_process_function(region)


def _tile_process_function(x: GenericNumberArray) -> GenericNumberArray:
    return np.asarray(np.argmax(x, axis=0).astype(np.uint8))


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
    logger.info("Writing TIFF %s", filename.with_suffix(".tiff"))
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
    logger.info("Done writing TIFF %s", filename.with_suffix(".tiff"))
