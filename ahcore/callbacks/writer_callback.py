from typing import Any

from pytorch_lightning import Callback

from ahcore.utils.io import get_logger
from multiprocessing import Process, Queue, Semaphore, Value
import ctypes
from threading import Thread
import time

logger = get_logger(__name__)


class WriterCallback(Callback):
    def __init__(self, queue_size: int = 16, max_concurrent_queues: int = 16):
        self._queue_size = queue_size
        self._queues = {}
        self._completion_flags = {}
        self._processes = {}

        self._semaphore = Semaphore(max_concurrent_queues)

        # Start a thread to monitor for process completion and cleanup
        self._cleanup_thread = Thread(target=self._monitor_and_cleanup)
        self._cleanup_thread.daemon = True  # Ensure the thread does not prevent the program from exiting
        self._cleanup_thread.start()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        indices = [i for i in range(len(batch["path"])) if i != 0 and batch["path"][i] != batch["path"][i - 1]]
        image_batches = [batch["image"][indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]
        logger.info(f"indices: {indices}")

        prev_filename = None
        last_batch = False
        for i, image_batch in enumerate(image_batches):
            if batch["path"][indices[i]] != prev_filename:
                last_batch = True
                prev_filename = batch["path"][indices[i]]
            self._process_batch(image_batch.cpu(), filename=batch["path"][indices[i]], last_batch=last_batch)

    def _process_batch(self, batch, filename, last_batch):
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
        self._queues[filename].put(batch)
        if last_batch:
            self._queues[filename].put(None)

    def _cleanup_completed_processes(self):
        for filename, flag in list(self._completion_flags.items()):
            if flag.value == 1:  # Process has completed
                logger.info(f"{filename} is completed.")
                process = self._processes[filename]
                process.join()  # Ensure process resources are freed
                # Cleanup queue and remove references
                del self._queues[filename]
                del self._processes[filename]
                del self._completion_flags[filename]
                logger.info(f"Cleaned up {filename}")

    def _monitor_and_cleanup(self):
        """Continuously monitor for completed processes and clean them up."""
        while True:
            self._cleanup_completed_processes()
            time.sleep(0.5)  # Short sleep to yield control and reduce CPU usage


# Mockup
class TileWriter:
    def __init__(self, filename):
        logger.info(f"Created tilewriter for {filename}")

    def consume(self, generator):
        for elem in generator:
            logger.info(f"Got {type(elem)}")


def queue_generator(queue):
    while True:
        item = queue.get()
        yield item
        if item is None:
            break


def writer_process(queue, filename, semaphore, completion_flag):
    try:
        writer = TileWriter(filename)
        writer.consume(queue_generator(queue))
    finally:
        completion_flag.value = 1
        semaphore.release()


# class WriterCallback(Callback):
#     # def __init__(self, writer_class, max_concurrent_queues=5):
#     def __init__(self, max_concurrent_queues=5):
#
#         # self._writer_class = writer_class
#         self._queues = {}
#         self._processes = {}
#         self._semaphore = Semaphore(max_concurrent_queues)
#
#         self._gather_results = True
#
#     def on_predict_batch_end(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         outputs: Any,
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int = 0,
#     ) -> None:
#
#         # We need to split the batch into a batch corresponding to a fixed ["path"]
#         #
#
#         logger.info(batch.keys())
#         logger.info(batch["filename"])
#
#         # identifier = self._extract_identifier_from_batch(batch)
#         # logger.info(f"Got {identifier} for batch {batch_idx} on {trainer.global_rank}")
#         #
#         # # Check if a new queue and process should be created for this filename
#         # if identifier not in self._queues:
#         #     logger.info(f"Got new {identifier} for batch {batch_idx} on {trainer.global_rank}")
#         #     self._semaphore.acquire()
#         #     logger.info(f"Acquired semaphore for new {identifier} for batch {batch_idx} on {trainer.global_rank}")
#         #     self._queues[identifier] = Queue()
#         #     process = Process(target=writer_process, args=(self._queues[identifier], identifier, self._semaphore))
#         #     self._processes[identifier] = process
#         #     process.start()
#         #
#         # # Add the batch to the queue for this filename
#         # self._queues[identifier].put(outputs)
#         #
#         # # Check if this is the last batch for this filename and signal the end if so
#         # # This is just in case
#         # if self._is_last_batch_for_file(identifier, batch_idx):
#         #     self._queues[identifier].put(None)
#
#     def on_inference_end(self, trainer, pl_module):
#         # Wait for all processes to finish
#         for process in self._processes.values():
#             process.join()
#
#     @staticmethod
#     def _extract_identifier_from_batch(batch):
#         return batch["filename"][0]
#
#     def _is_last_batch_for_file(self, filename, batch_idx):
#         # Implement logic to determine if this is the last batch for the given filename
#         is_last_batch = True
#         return is_last_batch
