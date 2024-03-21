import abc
import ctypes
import pathlib
import time
from multiprocessing import Event, Process, Queue, Semaphore, Value
from threading import Thread
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from dlup.data.dataset import ConcatDataset, TiledWsiDataset
from pytorch_lightning import Callback

from ahcore.utils.callbacks import sort_indices_row_major, sort_paths_and_return_both
from ahcore.utils.io import get_logger
from ahcore.utils.types import InferencePrecision, NormalizationType

logger = get_logger(__name__)


def _gather_batch(batch: dict, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Gather the batch from all processes in a distributed environment.
    This only gathers the coordinates, paths and the data as needed for the writers.

    Parameters
    ----------
    batch : dict
        The batch to gather.

    Returns
    -------
    tuple
        A tuple containing the gathered coordinates, data and paths.
    """
    if dist.is_initialized():
        world_size = dist.get_world_size()

        _coordinates = torch.stack(batch["coordinates"], dim=1)
        gathered_coords_list = [torch.zeros_like(_coordinates) for _ in range(world_size)]
        dist.all_gather(gathered_coords_list, _coordinates)
        all_coordinates = torch.cat(gathered_coords_list, dim=0)

        gathered_image_list = [torch.zeros_like(output) for _ in range(world_size)]
        dist.all_gather(gathered_image_list, output)
        all_data = torch.cat(gathered_image_list, dim=0)

        # Gather paths using all_gather_object for non-tensor data
        all_paths = [None] * world_size
        dist.all_gather_object(all_paths, batch["path"])
        all_paths = [path for sublist in all_paths for path in sublist]

    else:
        all_paths = batch["path"]
        all_coordinates = torch.stack(batch["coordinates"], dim=1)
        all_data = output

    return all_coordinates, all_data, all_paths


class WriterCallback(abc.ABC, Callback):
    def __init__(
        self,
        queue_size: int = 16,
        max_concurrent_queues: int = 16,
        requires_gather: bool = True,
        data_key: str = "prediction",
        normalization_type: str = NormalizationType.SOFTMAX,
        precision: str = InferencePrecision.FP32,  # This is passed to the writer class
        writer_class=None,
    ):
        # TODO: Test predict

        self._queue_size = queue_size
        self._queues: dict[str, Queue] = {}
        self._completion_flags = {}
        self._processes: dict[str, Process] = {}
        self._requires_gather = requires_gather
        self._data_key = data_key
        self._normalization_type = NormalizationType(normalization_type)
        self._precision: InferencePrecision = InferencePrecision(precision)
        self._writer_class = writer_class

        self._dataset_sizes: dict[str, int] = {}  # Keeps track of the total size of a dataset
        self._tile_counter: dict[str, int] = {}  # Keeps track of the number of tiles processed for a dataset

        self._semaphore = Semaphore(max_concurrent_queues)
        self._cleanup_shutdown_event = Event()

        self._total_dataset: ConcatDataset | None = None
        self._dataset_index = 0

    def _on_epoch_start(self, trainer: "pl.Trainer") -> None:
        self._cleanup_shutdown_event.clear()
        self._cleanup_thread = Thread(target=self._monitor_and_cleanup)
        self._cleanup_thread.daemon = True  # Ensure the thread does not prevent the program from exiting
        self._cleanup_thread.start()

        if self._dataset_sizes != {}:
            return

        current_dataset: TiledWsiDataset
        for current_dataset in self._total_dataset.datasets:
            self._dataset_sizes[current_dataset.slide_image.identifier] = len(current_dataset)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info("Validation epoch start")
        if trainer.global_rank != 0 and self._requires_gather:
            return

        self._total_dataset = trainer.datamodule.validate_dataset
        return self._on_epoch_start(trainer)

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info("Prediction epoch start")
        if trainer.global_rank != 0 and self._requires_gather:
            return
        self._total_dataset = trainer.predict_dataloaders.dataset
        return self._on_epoch_start(trainer)

    @staticmethod
    def _halt_for_val_or_sanity_limit(trainer: "pl.Trainer", batch_idx: int, last_batch: bool) -> bool:
        """During sanity checking or when the limit_val_batches is not 1, we may need to force the last batch to end.
        This is because the last batch might not be complete and we get stuck in the writer process."""
        if trainer.sanity_checking:
            total_steps = trainer.num_sanity_val_steps
            is_last_step = batch_idx + 1 == total_steps
        elif trainer.limit_val_batches != 1:
            if isinstance(trainer.limit_val_batches, int):
                total_steps = trainer.limit_val_batches
            else:
                total_steps = int(trainer.limit_val_batches * len(trainer.val_dataloaders))
            is_last_step = batch_idx + 1 == total_steps
        else:
            raise ValueError(
                "This function should only be called during sanity checking or when limit_val_batches is not 1."
            )
        if is_last_step:
            if not last_batch:
                logger.warning("Forcing last batch for writing -- slide might be incomplete!\n")
            return True
        elif last_batch:
            return True
        return False

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.world_size > 1 and self._requires_gather:  # Check if running in a distributed environment
            coordinates, data, paths = _gather_batch(batch, outputs[self._data_key])
        else:
            coordinates = torch.stack(batch["coordinates"], dim=1)
            data = outputs[self._data_key]
            paths = batch["path"]

        # You need to put this here *after* the gathering, otherwise it will hang!
        if trainer.global_rank != 0 and self._requires_gather:
            return

        paths, sorted_indices = sort_paths_and_return_both(paths)
        coordinates = coordinates[sorted_indices]
        data = data[sorted_indices]
        indices = [0] + [i for i in range(1, len(paths)) if paths[i] != paths[i - 1]] + [len(paths)]

        chunked_batch = [
            (
                paths[indices[idx] : indices[idx + 1]],
                coordinates[indices[idx] : indices[idx + 1]],
                data[indices[idx] : indices[idx + 1]],
            )
            for idx in range(len(indices) - 1)
        ]

        for i, (paths_chunk, coordinates, data) in enumerate(chunked_batch):
            # Need this sanity check at this point
            assert len(set(paths_chunk)) == 1
            curr_filename = paths_chunk[0]

            # We need to determine if this is the last batch for the current filename, so we can end the generator
            # there This might strictly not be needed if we can estimate the size of the grid, but it's a good safety
            # measure For instance, if we have multiresolution grids, or anything else that might make the standard grid
            # size estimation inaccurate
            if not self._tile_counter.get(curr_filename):
                self._tile_counter[curr_filename] = 0
            self._tile_counter[curr_filename] += data.shape[0]

            last_batch = False
            if self._tile_counter[curr_filename] == self._dataset_sizes[curr_filename]:
                last_batch = True
                self._tile_counter[curr_filename] = 0

            if trainer.sanity_checking or trainer.limit_val_batches != 1:
                last_batch = self._halt_for_val_or_sanity_limit(trainer, batch_idx, last_batch)

            coordinates = coordinates.cpu()
            data = data.cpu()
            row_major_indices = sort_indices_row_major(coordinates)
            coordinates = coordinates[row_major_indices].numpy()
            data = data[row_major_indices]
            data = NormalizationType.normalize(self._normalization_type)(data).detach().cpu().numpy()

            self._process_batch(
                coordinates,
                data,
                pl_module=pl_module,
                stage="validate",
                filename=curr_filename,
                last_batch=last_batch,
            )
            logger.info(f"Incrementing dataset index from {self._dataset_index} to {self._dataset_index + data.shape[0]}")
            self._dataset_index += data.shape[0]

    @abc.abstractmethod
    def build_writer_class(self, pl_module: "pl.LightningModule", stage: str, filename: pathlib.Path):
        pass

    def _process_batch(
        self,
        coordinates: torch.Tensor,
        batch: torch.Tensor,
        pl_module: "pl.LightningModule",
        stage: str,
        filename: str,
        last_batch: bool,
    ):
        if filename not in self._queues:
            self._semaphore.acquire()
            completion_flag = Value(ctypes.c_int, 0)  # For process completion signaling
            self._queues[filename] = Queue(maxsize=self._queue_size)
            process = Process(
                target=_writer_process,
                args=(self, self._queues[filename], filename, self._semaphore, completion_flag, stage, pl_module),
            )
            self._processes[filename] = process
            self._completion_flags[filename] = completion_flag
            process.start()

        self._queues[filename].put((coordinates, batch))
        if last_batch:
            self._queues[filename].put((None, None))

    def _cleanup_completed_processes(self):
        for filename, flag in list(self._completion_flags.items()):
            if flag.value == 1:  # Process has completed
                logger.debug(f"{filename} is completed. Clearing.")
                process = self._processes[filename]
                process.join()  # Ensure process resources are freed
                # Cleanup queue and remove references
                del self._queues[filename]
                del self._processes[filename]
                del self._completion_flags[filename]

    def _monitor_and_cleanup(self):
        """Continuously monitor for completed processes and clean them up."""
        while not self._cleanup_shutdown_event.is_set():
            self._cleanup_completed_processes()
            time.sleep(0.5)  # Short sleep to yield control and reduce CPU usage

    def _epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._dataset_index = 0
        while True:
            if self._queues == {}:
                break
            time.sleep(0.1)
        self._cleanup_shutdown_event.set()
        self._tile_counter = {}

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_end(trainer, pl_module)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_end(trainer, pl_module)


def _queue_generator(queue: Queue):
    """
    Generator to yield items from a queue. This is used to consume the queue in the writer process.

    Parameters
    ----------
    queue : Queue
        The queue to consume.

    Yields
    ------
    tuple
        A tuple containing the coordinates and the item (data, target, ...).

    """
    while True:
        coordinates, item = queue.get()
        if item is None:
            break
        yield coordinates, item


def _writer_process(
    callback_instance,
    queue: Queue,
    filename: str,
    semaphore: Semaphore,
    completion_flag: Value,
    stage: str,
    pl_module: "pl.LightningModule",
):
    """
    Process to consume a queue and write to a writer.

    Parameters
    ----------
    queue : Queue
        The queue to consume.
    filename : str
        The filename to write to.
    semaphore : Semaphore
        The semaphore to release when the process is done.
    completion_flag : Value
        The flag to signal process completion.

    Returns
    ------
    None
    """
    try:
        writer = callback_instance.build_writer_class(pl_module, stage, filename)
        writer.consume(_queue_generator(queue))
        logger.debug(f"Stopped writing for {filename}")
    except Exception as e:
        logger.exception(f"Error in writer_process for {filename}: {e}")
    finally:
        completion_flag.value = 1
        semaphore.release()
