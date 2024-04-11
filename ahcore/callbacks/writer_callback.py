import abc
import ctypes
import pathlib
import time
from multiprocessing import Event, Process, Queue, Semaphore, Value
from multiprocessing.sharedctypes import SynchronizedBase
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Semaphore as SemaphoreClass
from threading import Thread
from typing import Any, Generator, Tuple

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from dlup.data.dataset import ConcatDataset, TiledWsiDataset
from pytorch_lightning import Callback

from ahcore.callbacks.converters.tiff_callback import ConvertCallbacks
from ahcore.lit_module import AhCoreLightningModule
from ahcore.utils.io import get_logger
from ahcore.utils.types import GenericNumberArray, InferencePrecision, NormalizationType
from ahcore.writers import Writer

logger = get_logger(__name__)


def _remove_initial_gap_and_zeros(tensor: torch.Tensor, gap_threshold: int = 1) -> int:
    """Function to detect if there are initial zeros appended after the gather."""
    if len(tensor) < 2:
        return 0

    # TODO BETTER CHECK!!!!
    # if (tensor[-1] - tensor[0]) >= len(tensor):
    #     return 0

    start_index = 0

    # Remove leading zeros
    while start_index < len(tensor) and tensor[start_index] == 0:
        if len(tensor) == start_index + 1:
            break
        if tensor[start_index + 1] == 1:
            break
        start_index += 1

    # Now find the gap, if any, after the leading zeros
    for i in range(start_index, len(tensor) - 1):
        if tensor[i + 1] - tensor[i] > gap_threshold:
            start_index = i + 1
            break

    return start_index


def _gather_batch(
    batch: dict[str, Any], output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
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
        _global_index = batch["global_index"]
        _gathered_global_index = [torch.zeros_like(_global_index) for _ in range(world_size)]
        dist.all_gather(_gathered_global_index, _global_index)
        all_global_index = torch.cat(_gathered_global_index, dim=0)
        all_global_index, sorting_index = torch.sort(all_global_index)

        gathered_coords_list = [torch.zeros_like(_coordinates) for _ in range(world_size)]
        dist.all_gather(gathered_coords_list, _coordinates)
        all_coordinates = torch.cat(gathered_coords_list, dim=0)[sorting_index]

        gathered_image_list = [torch.zeros_like(output) for _ in range(world_size)]
        dist.all_gather(gathered_image_list, output)
        all_data = torch.cat(gathered_image_list, dim=0)[sorting_index]

        # Gather paths using all_gather_object for non-tensor data
        all_paths = [""] * world_size
        dist.all_gather_object(all_paths, batch["path"])
        all_paths = [path for sublist in all_paths for path in sublist]
        all_paths = [all_paths[i] for i in sorting_index.cpu().numpy().tolist()]

    else:
        all_paths = batch["path"]
        all_coordinates = torch.stack(batch["coordinates"], dim=1)
        all_data = output
        all_global_index = batch["global_index"]

    return all_coordinates, all_data, all_global_index, all_paths


class WriterCallback(abc.ABC, Callback):
    def __init__(
        self,
        writer_class: Writer,
        dump_dir: pathlib.Path,
        queue_size: int = 16,
        max_concurrent_queues: int = 16,
        requires_gather: bool = True,
        data_key: str = "prediction",
        normalization_type: str = NormalizationType.SOFTMAX,
        precision: str = InferencePrecision.FP32,  # This is passed to the writer class
        callbacks: list[ConvertCallbacks] | None = None,
    ) -> None:
        # TODO: Test predict
        self._dump_dir = dump_dir
        self._queue_size = queue_size
        self._queues: dict[str, Queue[Tuple[GenericNumberArray | None, GenericNumberArray | None]]] = {}
        self._completion_flags: dict[str, Any] = {}
        self._processes: dict[str, Process] = {}
        self._requires_gather = requires_gather
        self._data_key = data_key
        self._normalization_type = NormalizationType(normalization_type)
        self._precision: InferencePrecision = InferencePrecision(precision)
        self._writer_class = writer_class
        self._max_concurrent_queues = max_concurrent_queues
        self._callbacks = callbacks or []

        self._dataset_sizes: dict[str, int] = {}  # Keeps track of the total size of a dataset
        self._tile_counter: dict[str, int] = {}  # Keeps track of the number of tiles processed for a dataset

        self._semaphore = Semaphore(max_concurrent_queues)
        self._cleanup_shutdown_event: EventClass = Event()

        self._total_dataset: ConcatDataset[dict[str, Any]] | None = None
        self._dataset_index = 0

    @property
    def dump_dir(self) -> pathlib.Path:
        return self._dump_dir

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if trainer.world_size > 1:
            if self._max_concurrent_queues < trainer.world_size:
                raise ValueError(
                    "max_concurrent_queues should be greater than or equal to the world size to avoid deadlock."
                )

        # TODO: is this only required in rank 0?
        for callback in self._callbacks:
            callback.setup(self, trainer, pl_module, stage)

    @abc.abstractmethod
    def get_output_filename(self, pl_module: AhCoreLightningModule, filename: str) -> pathlib.Path:
        pass

    def _on_epoch_start(self, trainer: "pl.Trainer") -> None:
        self._cleanup_shutdown_event.clear()
        self._cleanup_thread = Thread(target=self._monitor_and_cleanup)
        self._cleanup_thread.daemon = True  # Ensure the thread does not prevent the program from exiting
        self._cleanup_thread.start()

        if self._dataset_sizes != {}:
            return

        current_dataset: TiledWsiDataset
        assert self._total_dataset
        for current_dataset in self._total_dataset.datasets:  # type: ignore
            assert current_dataset.slide_image.identifier
            self._dataset_sizes[current_dataset.slide_image.identifier] = len(current_dataset)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.debug("Validation epoch start")
        if trainer.global_rank != 0 and self._requires_gather:
            return
        self._total_dataset = trainer.datamodule.validate_dataset  # type: ignore
        return self._on_epoch_start(trainer)

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info("Prediction epoch start")
        if trainer.global_rank != 0 and self._requires_gather:
            return
        self._total_dataset = trainer.predict_dataloaders.dataset  # type: ignore
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
                assert trainer.val_dataloaders
                assert trainer.limit_val_batches
                total_steps = int(trainer.limit_val_batches) * len(trainer.val_dataloaders)
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
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, "validate", dataloader_idx)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, "predict", dataloader_idx)

    def _batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        stage: str,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.world_size > 1 and self._requires_gather:  # Check if running in a distributed environment
            coordinates, data, global_indices, paths = _gather_batch(batch, outputs[self._data_key])
            # TODO: Make more efficient
            start_index = _remove_initial_gap_and_zeros(global_indices)
            coordinates = coordinates[start_index:]
            data = data[start_index:]
            paths = paths[start_index:]
        else:
            coordinates = torch.stack(batch["coordinates"], dim=1)
            data = outputs[self._data_key]
            paths = batch["path"]

        # You need to put this here *after* the gathering, otherwise it will hang!
        if trainer.global_rank != 0 and self._requires_gather:
            return

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

            data = NormalizationType.normalize(self._normalization_type)(data).detach()

            self._process_batch(
                coordinates.cpu().numpy(),
                data.cpu().numpy(),
                pl_module=pl_module,
                stage=stage,
                filename=curr_filename,
                last_batch=last_batch,
            )
            self._dataset_index += data.shape[0]

            if last_batch:
                logger.info(f"Processed {curr_filename}, now dumping.")

    @abc.abstractmethod
    def build_writer_class(self, pl_module: "pl.LightningModule", stage: str, filename: str) -> Writer:
        pass

    def _process_batch(
        self,
        coordinates: GenericNumberArray,
        batch: GenericNumberArray,
        pl_module: "pl.LightningModule",
        stage: str,
        filename: str,
        last_batch: bool,
    ) -> None:
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

    def _cleanup_completed_processes(self) -> None:
        for filename, flag in list(self._completion_flags.items()):
            if flag.value == 1:  # Process has completed
                logger.debug(f"{filename} is completed. Clearing.")
                process = self._processes[filename]
                process.join()  # Ensure process resources are freed
                # Cleanup queue and remove references
                del self._queues[filename]
                del self._processes[filename]
                del self._completion_flags[filename]

    def _monitor_and_cleanup(self) -> None:
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

    def start_callbacks(self, filename: str) -> None:
        for callback in self._callbacks:
            callback.start(filename)


def _queue_generator(
    queue: "Queue[Tuple[GenericNumberArray | None, GenericNumberArray | None]]",
) -> Generator[Tuple[GenericNumberArray, GenericNumberArray], None, None]:
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
        if item is None or coordinates is None:
            break
        yield coordinates, item


def _writer_process(
    callback_instance: WriterCallback,
    queue: "Queue[Tuple[GenericNumberArray | None, GenericNumberArray | None]]",
    filename: str,
    semaphore: SemaphoreClass,
    completion_flag: "SynchronizedBase[ctypes.c_int]",
    stage: str,
    pl_module: "pl.LightningModule",
) -> None:
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
        callback_instance.start_callbacks(filename)

    except Exception as e:
        logger.exception(f"Error in writer_process for {filename}: {e}")
    finally:
        completion_flag.value = 1  # type: ignore
        semaphore.release()
