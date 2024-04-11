from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
from typing import Any, Optional, cast

import pytorch_lightning as pl
from pytorch_lightning import Callback

from ahcore.callbacks import WriteFileCallback
from ahcore.callbacks.converters.tiff_callback import _generator_from_reader, _tile_process_function, _write_tiff
from ahcore.lit_module import AhCoreLightningModule
from ahcore.readers import FileImageReader
from ahcore.utils.callbacks import get_output_filename
from ahcore.utils.io import get_logger

logger = get_logger(__name__)


class WriteTiffCallback(Callback):
    def __init__(
        self,
        reader_class: FileImageReader,
        max_concurrent_writers: int,
        tile_size: tuple[int, int] = (1024, 1024),
        colormap: dict[int, str] | None = None,
        max_patience_on_unfinished_files: int = 100,
    ):
        self._reader_class = reader_class
        self._pool = multiprocessing.Pool(max_concurrent_writers)
        self._dump_dir: Optional[Path] = None

        self._model_name: str | None = None
        self._tile_size = tile_size
        self._colormap = colormap
        self._patience_on_unfinished_files = max_patience_on_unfinished_files

        # TODO: Handle tile operation such that we avoid repetitions.

        self._tile_process_function = _tile_process_function  # function that is applied to the tile.
        self._filenames: dict[Path, Path] = {}  # This has all the cache files

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

        _callback: Optional[WriteFileCallback] = None
        for idx, callback in enumerate(trainer.callbacks):  # type: ignore
            if isinstance(callback, WriteFileCallback):
                _callback = cast(WriteFileCallback, trainer.callbacks[idx])  # type: ignore
                break
        if _callback is None:
            raise ValueError("WriteFileCallback required before tiff images can be written using this Callback.")

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
                output_filename = get_output_filename(
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
        logger.info("Writing TIFF files to %s", self.dump_dir / "outputs" / f"{pl_module.name}")

        results = []
        unfinished_files = []  # Store partial files for retry

        for image_filename, cache_filename in self._filenames.items():
            if not cache_filename.exists():
                unfinished_files.append((image_filename, cache_filename))
                continue

            logger.debug(
                "Writing image output %s to %s", Path(image_filename), Path(image_filename).with_suffix(".tiff")
            )
            output_path = self.dump_dir / "outputs" / f"{pl_module.name}" / f"step_{pl_module.global_step}"
            with open(output_path / "image_tiff_link.txt", "a") as file:
                file.write(f"{image_filename},{cache_filename.with_suffix('.tiff')}\n")

            result = self._pool.apply_async(
                _write_tiff,
                (
                    cache_filename,
                    self._tile_size,
                    self._tile_process_function,
                    self._colormap,
                    self._reader_class,
                    _generator_from_reader,
                ),
            )
            results.append(result)

        # Retry mechanism for unfinished files
        iterations = 0
        while unfinished_files:
            retry_files = []
            for image_filename, cache_filename in unfinished_files:
                if cache_filename.exists():  # Check if ready
                    logger.debug("Retrying to write TIFF %s", cache_filename.with_suffix(".tiff"))
                    result = self._pool.apply_async(
                        _write_tiff,
                        (
                            cache_filename,
                            self._tile_size,
                            self._tile_process_function,
                            self._colormap,
                            self._reader_class,
                            _generator_from_reader,
                        ),
                    )
                    results.append(result)
                else:
                    logger.debug("Cache file %s is still unfinished. Placing back in queue", cache_filename)
                    retry_files.append((image_filename, cache_filename))
            unfinished_files = retry_files
            if unfinished_files:
                iterations += 1
                if iterations > self._patience_on_unfinished_files:
                    raise ValueError(f"Failed to write TIFF files {unfinished_files} after {iterations} iterations.")
                time.sleep(2.0)  # Wait a bit before retrying

        for result in results:
            result.get()  # Ensure all retry tasks are also completed

        self._filenames = {}  # Reset the filenames for the next epoch

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


