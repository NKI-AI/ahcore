from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Optional, cast

import numpy as np
import pytorch_lightning as pl
from dlup._image import Resampling
from dlup.writers import TiffCompression, TifffileImageWriter
from numpy import typing as npt
from pytorch_lightning import Callback

from ahcore.callbacks import WriteH5Callback
from ahcore.lit_module import AhCoreLightningModule
from ahcore.readers import H5FileImageReader, StitchingMode
from ahcore.utils.callbacks import _ValidationDataset, get_h5_output_filename
from ahcore.utils.io import get_logger
from ahcore.utils.types import GenericNumberArray

logger = get_logger(__name__)


class WriteTiffCallback(Callback):
    def __init__(
        self,
        max_concurrent_writers: int,
        tile_size: tuple[int, int] = (1024, 1024),
        colormap: dict[int, str] | None = None,
    ):
        self._pool = multiprocessing.Pool(max_concurrent_writers)
        self._logger = get_logger(type(self).__name__)
        self._dump_dir: Optional[Path] = None

        self._model_name: str | None = None
        self._tile_size = tile_size
        self._colormap = colormap

        # TODO: Handle tile operation such that we avoid repetitions.

        self._tile_process_function = _tile_process_function  # function that is applied to the tile.
        self._filenames: dict[Path, Path] = {}  # This has all the h5 files

    @property
    def dump_dir(self) -> Optional[Path]:
        return self._dump_dir

    def _validate_parameters(self) -> None:
        dump_dir = self._dump_dir
        if not dump_dir:
            raise ValueError("Dump directory is not set.")

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        if not isinstance(pl_module, AhCoreLightningModule):
            # TODO: Make a AhCoreCallback with these features
            raise ValueError("AhCoreLightningModule required for WriteTiffCallback.")

        self._model_name = pl_module.name

        _callback: Optional[WriteH5Callback] = None
        for idx, callback in enumerate(trainer.callbacks):  # type: ignore
            if isinstance(callback, WriteH5Callback):
                _callback = cast(WriteH5Callback, trainer.callbacks[idx])  # type: ignore
                break
        if _callback is None:
            raise ValueError("WriteH5Callback required before tiff images can be written using this Callback.")

        # This is needed for mypy
        assert _callback, "_callback should never be None after the setup."
        assert _callback.dump_dir, "_callback.dump_dir should never be None after the setup."
        self._dump_dir = _callback.dump_dir

    def _batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.global_rank != 0:
            return
        assert self.dump_dir, "dump_dir should never be None here."

        filenames = set([Path(_) for _ in batch["path"]])
        for filename in filenames:
            if filename not in self._filenames:
                output_filename = get_h5_output_filename(
                    dump_dir=self.dump_dir,
                    input_path=filename,
                    model_name=str(pl_module.name),
                    step=pl_module.global_step,
                )
                self._filenames[filename] = output_filename

    def _epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        assert self.dump_dir, "dump_dir should never be None here."
        self._logger.info("Writing TIFF files to %s", self.dump_dir / "outputs" / f"{pl_module.name}")
        results = []
        for image_filename, h5_filename in self._filenames.items():
            self._logger.debug(
                "Writing image output %s to %s",
                Path(image_filename),
                Path(image_filename).with_suffix(".tiff"),
            )
            output_path = self.dump_dir / "outputs" / f"{pl_module.name}" / f"step_{pl_module.global_step}"
            with open(output_path / "image_tiff_link.txt", "a") as file:
                file.write(f"{image_filename},{h5_filename.with_suffix('.tiff')}\n")
            if not h5_filename.exists():
                self._logger.warning("H5 file %s does not exist. Skipping", h5_filename)
                continue

            result = self._pool.apply_async(
                _write_tiff,
                (
                    h5_filename,
                    self._tile_size,
                    self._tile_process_function,
                    self._colormap,
                    _generator_from_reader,
                ),
            )
            results.append(result)

        for result in results:
            result.get()  # Wait for the process to complete.
        self._filenames = {}  # Reset the filenames

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_end(trainer, pl_module)

    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_end(trainer, pl_module)


def _generator_from_reader(
    h5_reader: H5FileImageReader,
    tile_size: tuple[int, int],
    tile_process_function: Callable[[GenericNumberArray], GenericNumberArray],
) -> Generator[GenericNumberArray, None, None]:
    validation_dataset = _ValidationDataset(
        data_description=None,
        native_mpp=h5_reader.mpp,
        reader=h5_reader,
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
    generator_from_reader: Callable[
        [H5FileImageReader, tuple[int, int], Callable[[GenericNumberArray], GenericNumberArray]],
        Iterator[npt.NDArray[np.int_]],
    ],
) -> None:
    logger.debug("Writing TIFF %s", filename.with_suffix(".tiff"))
    with H5FileImageReader(filename, stitching_mode=StitchingMode.CROP) as h5_reader:
        writer = TifffileImageWriter(
            filename.with_suffix(".tiff"),
            size=h5_reader.size,
            mpp=h5_reader.mpp,
            tile_size=tile_size,
            pyramid=True,
            compression=TiffCompression.ZSTD,
            quality=100,
            interpolator=Resampling.NEAREST,
            colormap=colormap,
        )
        writer.from_tiles_iterator(generator_from_reader(h5_reader, tile_size, tile_process_function))
