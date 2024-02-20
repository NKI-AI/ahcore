import ctypes
import time
from multiprocessing import Process, Queue, Semaphore, Value
from pprint import pformat
from threading import Thread
from typing import Any

import pytorch_lightning as pl
import torch
from dlup.data.dataset import ConcatDataset, TiledWsiDataset
from pytorch_lightning import Callback
import torch.distributed as dist

from ahcore.utils.io import get_logger

logger = get_logger(__name__)


def _gather_batch(batch, data_key: str):
    if dist.is_initialized():
        world_size = dist.get_world_size()

        # Stacking coordinates as a single tensor
        _coordinates = torch.stack(batch["coordinates"], dim=1)

        # Prepare tensors for gathering across all processes
        gathered_coords_list = [torch.zeros_like(_coordinates) for _ in range(world_size)]
        # Gather coordinates
        dist.all_gather(gathered_coords_list, _coordinates)

        # Concatenate gathered coordinates along the first dimension (now they are in a list)
        all_coordinates = torch.cat(gathered_coords_list, dim=0)

        # Gather paths using all_gather_object for non-tensor data
        all_paths = [None] * world_size
        dist.all_gather_object(all_paths, batch["path"])
        all_paths = [path for sublist in all_paths for path in sublist]

        gathered_image_list = [torch.zeros_like(batch[data_key]) for _ in range(world_size)]
        dist.all_gather(gathered_image_list, batch[data_key])
        all_data = torch.cat(gathered_image_list, dim=0)

    else:
        all_paths = batch["path"]
        all_coordinates = torch.stack(batch["coordinates"], dim=1)
        all_data = batch[data_key]

    return all_coordinates, all_data, all_paths

class WriterCallback(Callback):
    def __init__(self, queue_size: int = 16, max_concurrent_queues: int = 16, requires_gather: bool = True):
        # TODO: Test semaphores
        # TODO: Test cleanup
        # TODO: Test multigpu
        # TODO: Test predict
        # TODO: Make flag variable.
        # TODO: If multigpu and gather required, skip the other ranks

        self._queue_size = queue_size
        self._queues: dict[str, Queue] = {}
        self._completion_flags = {}
        self._processes: dict[str, Process] = {}
        self._requires_gather = requires_gather

        self._dataset_index = 0  # Keeps track of the current index in the dataset
        self._dataset_sizes: dict[str, int] = {}  # Keeps track of the total size of a dataset
        self._tile_counter: dict[str, int] = {}  # Keeps track of the number of tiles processed for a dataset

        self._semaphore = Semaphore(max_concurrent_queues)

        # Start a thread to monitor for process completion and cleanup
        self._cleanup_thread = Thread(target=self._monitor_and_cleanup)
        self._cleanup_thread.daemon = True  # Ensure the thread does not prevent the program from exiting
        self._cleanup_thread.start()

    def on_validation_epoch_start(self, trainer, pl_module):
        if self._dataset_sizes != {}:
            return

        total_dataset: ConcatDataset = trainer.datamodule.validate_dataset  # type: ignore
        current_dataset: TiledWsiDataset

        for current_dataset in total_dataset.datasets:
            self._dataset_sizes[current_dataset.slide_image.identifier] = len(current_dataset)

        logger.info(f"Dataset sizes: {pformat(self._dataset_sizes)}")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        data_key = "image"

        if trainer.world_size > 1 and self._requires_gather:  # Check if running in a distributed environment
            coordinates, data, paths = _gather_batch(batch, data_key)
        else:
            coordinates = torch.stack(batch["coordinates"], dim=1)
            data = batch[data_key]
            paths = batch["path"]

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
            logger.info(f"{filename} not in queue")
            self._semaphore.acquire()
            logger.info("Acquired semaphore")
            completion_flag = Value(ctypes.c_int, 0)  # For process completion signaling
            self._queues[filename] = Queue(maxsize=self._queue_size)
            process = Process(
                target=writer_process, args=(self._queues[filename], filename, self._semaphore, completion_flag)
            )
            self._processes[filename] = process
            self._completion_flags[filename] = completion_flag
            process.start()
        logger.info(f"Putting {filename} in queue {type(batch)}")
        self._queues[filename].put((coordinates, batch))
        if last_batch:
            self._queues[filename].put((None, None))

    def _cleanup_completed_processes(self):
        for filename, flag in list(self._completion_flags.items()):
            if flag.value == 1:  # Process has completed
                # logger.info(f"{filename} is completed.")
                process = self._processes[filename]
                process.join()  # Ensure process resources are freed
                # Cleanup queue and remove references
                del self._queues[filename]
                del self._processes[filename]
                del self._completion_flags[filename]
                # logger.info(f"Cleaned up {filename}")

    def _monitor_and_cleanup(self):
        """Continuously monitor for completed processes and clean them up."""
        while True:
            self._cleanup_completed_processes()
            time.sleep(0.5)  # Short sleep to yield control and reduce CPU usage

    # TODO: This needs to be properly called.
    def _epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._dataset_index = 0

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_end(trainer, pl_module)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_end(trainer, pl_module)


# Mockup
class BasicTileWriter:
    def __init__(self, filename):
        self._filename = filename
        logger.info(f"Created BasicTileWriter for {filename}")
        self._counter = 0

    def consume(self, generator):
        for coordinates, item in generator:
            # TODO: We need to unroll the batch here
            for coordinates_slice, item_slice in zip(coordinates, item):
                self._counter += 1
                print(f"{self._counter}: Got coordinates {coordinates_slice} in {self._filename}")


def queue_generator(queue):
    while True:
        coordinates, item = queue.get()
        if item is None:
            break
        yield coordinates, item


def writer_process(queue, filename, semaphore, completion_flag):
    try:
        writer = BasicTileWriter(filename)
        writer.consume(queue_generator(queue))
        logger.info(f"Stopped writing for {filename}")
    except Exception as e:
        logger.exception(f"Error in writer_process for {filename}: {e}")
    finally:
        completion_flag.value = 1
        semaphore.release()

#