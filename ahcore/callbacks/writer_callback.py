import ctypes
import time
from multiprocessing import Process, Queue, Semaphore, Value
from threading import Thread
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from dlup.data.dataset import ConcatDataset, TiledWsiDataset
from pytorch_lightning import Callback
from functools import partial
from ahcore.utils.io import get_logger

logger = get_logger(__name__)


class BasicTileWriter:
    def __init__(self, filename):
        self._filename = filename
        logger.debug(f"Created BasicTileWriter for {filename}")
        self._counter = 0

    def consume(self, generator):
        for coordinates, item in generator:
            for coordinates_slice, item_slice in zip(coordinates, item):
                self._counter += 1
                logger.debug(f"{self._counter}: Got coordinates {coordinates_slice} in {self._filename}")


def _gather_batch(batch: dict, data_key: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
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

        gathered_image_list = [torch.zeros_like(batch[data_key]) for _ in range(world_size)]
        dist.all_gather(gathered_image_list, batch[data_key])
        all_data = torch.cat(gathered_image_list, dim=0)

        # Gather paths using all_gather_object for non-tensor data
        all_paths = [None] * world_size
        dist.all_gather_object(all_paths, batch["path"])
        all_paths = [path for sublist in all_paths for path in sublist]

    else:
        all_paths = batch["path"]
        all_coordinates = torch.stack(batch["coordinates"], dim=1)
        all_data = batch[data_key]

    return all_coordinates, all_data, all_paths


class WriterCallback(Callback):
    def __init__(
        self,
        queue_size: int = 16,
        max_concurrent_queues: int = 16,
        requires_gather: bool = True,
        data_key: str = "image",
        writer_class=BasicTileWriter,
    ):
        # TODO: Test predict

        self._queue_size = queue_size
        self._queues: dict[str, Queue] = {}
        self._completion_flags = {}
        self._processes: dict[str, Process] = {}
        self._requires_gather = requires_gather
        self._data_key = data_key
        self._writer_class = writer_class

        self._dataset_sizes: dict[str, int] = {}  # Keeps track of the total size of a dataset
        self._tile_counter: dict[str, int] = {}  # Keeps track of the number of tiles processed for a dataset

        self._semaphore = Semaphore(max_concurrent_queues)

        # Start a thread to monitor for process completion and cleanup
        self._cleanup_thread: Thread | None = Thread(target=self._monitor_and_cleanup)
        self._cleanup_thread.daemon = True  # Ensure the thread does not prevent the program from exiting
        self._cleanup_thread.start()

    def _on_epoch_start(self, trainer: "pl.Trainer", total_dataset: ConcatDataset) -> None:
        if self._dataset_sizes != {}:
            return

        current_dataset: TiledWsiDataset

        for current_dataset in total_dataset.datasets:
            self._dataset_sizes[current_dataset.slide_image.identifier] = len(current_dataset)

        if trainer.global_rank != 0 and self._requires_gather:
            self._cleanup_thread = Thread(target=self._monitor_and_cleanup)
            self._cleanup_thread.daemon = True  # Ensure the thread does not prevent the program from exiting
            self._cleanup_thread.start()

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info("Validation epoch start")
        if trainer.global_rank != 0 and self._requires_gather:
            return
        return self._on_epoch_start(trainer, trainer.datamodule.validate_dataset)

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info("Prediction epoch start")
        if trainer.global_rank != 0 and self._requires_gather:
            return
        return self._on_epoch_start(trainer, trainer.predict_dataloaders.dataset)

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
            coordinates, data, paths = _gather_batch(batch, self._data_key)
        else:
            coordinates = torch.stack(batch["coordinates"], dim=1)
            data = batch[self._data_key]
            paths = batch["path"]

        # You need to put this here *after* the gathering, otherwise it will hang!
        if trainer.global_rank != 0 and self._requires_gather:
            return

        indices = [0] + [i for i in range(len(paths)) if paths[i] != paths[i - 1]] + [len(paths)]
        chunked_batch = [
            (coordinates[indices[idx] : indices[idx + 1]], data[indices[idx] : indices[idx + 1]])
            for idx in range(len(indices) - 1)
        ]

        for i, (coordinates, data) in enumerate(chunked_batch):
            curr_filename = paths[indices[i]]
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

            self._process_batch(
                coordinates.cpu().numpy(), data.cpu().numpy(), filename=curr_filename, last_batch=last_batch
            )

    def _process_batch(self, coordinates: torch.Tensor, batch: torch.Tensor, filename: str, last_batch: bool):
        if filename not in self._queues:
            logger.debug(f"{filename} not in queue")
            self._semaphore.acquire()
            logger.debug("Acquired semaphore")
            completion_flag = Value(ctypes.c_int, 0)  # For process completion signaling
            self._queues[filename] = Queue(maxsize=self._queue_size)
            process = Process(
                target=partial(_writer_process, writer=self._writer_class), args=(self._queues[filename], filename, self._semaphore, completion_flag)
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
        while True:
            self._cleanup_completed_processes()
            time.sleep(0.5)  # Short sleep to yield control and reduce CPU usage

    def _epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        while True:
            if self._queues == {}:
                break
            time.sleep(0.1)
        del self._cleanup_thread

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


def _writer_process(queue: Queue, filename: str, semaphore: Semaphore, completion_flag: Value, writer):
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
        _writer = writer(filename)
        _writer.consume(_queue_generator(queue))
        logger.debug(f"Stopped writing for {filename}")
    except Exception as e:
        logger.exception(f"Error in writer_process for {filename}: {e}")
    finally:
        completion_flag.value = 1
        semaphore.release()
