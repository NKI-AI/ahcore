from __future__ import annotations

from multiprocessing import Pipe, Process, Queue, Semaphore
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Generator, Optional, TypedDict

import pytorch_lightning as pl
import torch
from dlup.data.dataset import ConcatDataset, TiledWsiDataset
from pytorch_lightning import Callback

from ahcore.callbacks.utils import _get_h5_output_filename
from ahcore.utils.data import DataDescription, GridDescription
from ahcore.utils.io import get_logger
from ahcore.utils.types import GenericArray
from ahcore.writers import H5FileImageWriter


class WriteH5Callback(Callback):
    def __init__(self, max_queue_size: int, max_concurrent_writers: int, dump_dir: Path):
        """
        Callback to write predictions to H5 files. This callback is used to write whole-slide predictions to single H5
        files in a separate thread.

        TODO:
            - Add support for distributed data parallel

        Parameters
        ----------
        max_queue_size : int
            The maximum number of items to store in the queue (i.e. tiles).
        max_concurrent_writers : int
            The maximum number of concurrent writers.
        dump_dir : pathlib.Path
            The directory to dump the H5 files to.
        """
        super().__init__()
        self._writers: dict[str, _WriterMessage] = {}
        self._current_filename = None
        self._dump_dir = Path(dump_dir)
        self._max_queue_size = max_queue_size
        self._semaphore = Semaphore(max_concurrent_writers)
        self._dataset_index = 0

        self._logger = get_logger(type(self).__name__)

    @property
    def dump_dir(self) -> Path:
        return self._dump_dir

    def __process_management(self) -> None:
        """
        Handle the graceful termination of multiple processes at the end of h5 writing.
        This block ensures proper release of resources allocated during multiprocessing.

        Returns
        -------
        None
        """
        assert self._current_filename, "_current_filename shouldn't be None here"

        self._writers[self._current_filename]["queue"].put(None)
        self._writers[self._current_filename]["process"].join()
        self._writers[self._current_filename]["process"].close()
        self._writers[self._current_filename]["queue"].close()

    @property
    def writers(self) -> dict[str, _WriterMessage]:
        return self._writers

    def _batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        stage: str,
        dataloader_idx: int = 0,
    ) -> None:
        filename = batch["path"][0]  # Filenames are constant across the batch.
        if any([filename != path for path in batch["path"]]):
            raise ValueError(
                "All paths in a batch must be the same. "
                "Either use batch_size=1 or ahcore.data.samplers.WsiBatchSampler."
            )

        if filename != self._current_filename:
            output_filename = _get_h5_output_filename(
                self.dump_dir,
                filename,
                model_name=str(pl_module.name),
                step=pl_module.global_step,
            )
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            link_fn = (
                self.dump_dir / "outputs" / f"{pl_module.name}" / f"step_{pl_module.global_step}" / "image_h5_link.txt"
            )
            with open(link_fn, "a" if link_fn.is_file() else "w") as file:
                file.write(f"{filename},{output_filename}\n")

            self._logger.debug("%s -> %s", filename, output_filename)
            if self._current_filename is not None:
                self.__process_management()
                self._semaphore.release()

            self._semaphore.acquire()

            if stage == "validate":
                total_dataset: ConcatDataset = trainer.datamodule.validate_dataset  # type: ignore
            elif stage == "predict":
                total_dataset: ConcatDataset = trainer.predict_dataloaders.dataset  # type: ignore
            else:
                raise NotImplementedError(f"Stage {stage} is not supported for {self.__class__.__name__}.")

            current_dataset: TiledWsiDataset
            current_dataset, _ = total_dataset.index_to_dataset(self._dataset_index)  # type: ignore
            slide_image = current_dataset.slide_image

            data_description: DataDescription = pl_module.data_description  # type: ignore
            inference_grid: GridDescription = data_description.inference_grid

            mpp = inference_grid.mpp
            if mpp is None:
                mpp = slide_image.mpp

            size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))
            num_samples = len(current_dataset)

            # Let's get the data_description, so we can figure out the tile size and things like that
            tile_size = inference_grid.tile_size
            tile_overlap = inference_grid.tile_overlap

            # TODO: We are really putting strange things in the Queue if we may believe mypy
            new_queue: Queue[Any] = Queue()  # pylint: disable=unsubscriptable-object
            parent_conn, child_conn = Pipe()
            new_writer = H5FileImageWriter(
                output_filename,
                size=size,
                mpp=mpp,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                num_samples=num_samples,
                progress=None,
            )
            new_process = Process(target=new_writer.consume, args=(self.generator(new_queue), child_conn))
            new_process.start()
            self._writers[filename] = {
                "queue": new_queue,
                "writer": new_writer,
                "process": new_process,
                "connection": parent_conn,
            }
            self._current_filename = filename

        prediction = outputs["prediction"].detach().cpu().numpy()
        coordinates_x, coordinates_y = batch["coordinates"]
        coordinates = torch.stack([coordinates_x, coordinates_y]).T.detach().cpu().numpy()
        self._writers[filename]["queue"].put((coordinates, prediction))
        self._dataset_index += prediction.shape[0]

    def _epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._current_filename is not None:
            self.__process_management()
            self._semaphore.release()
            self._dataset_index = 0
        # Reset current filename to None for correct execution of subsequent validation loop
        self._current_filename = None
        # Clear all the writers from the current epoch
        self._writers = {}

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, "validate", dataloader_idx)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, "predict", dataloader_idx)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_end(trainer, pl_module)

    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_end(trainer, pl_module)

    @staticmethod
    def generator(
        queue: Queue[Optional[GenericArray]],  # pylint: disable=unsubscriptable-object
    ) -> Generator[GenericArray, None, None]:
        while True:
            batch = queue.get()
            if batch is None:
                break
            yield batch


class _WriterMessage(TypedDict):
    queue: Queue[Optional[tuple[GenericArray, GenericArray]]]  # pylint: disable=unsubscriptable-object
    writer: H5FileImageWriter
    process: Process
    connection: Connection
