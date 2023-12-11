from multiprocessing import Process, Queue, Semaphore
from typing import Any

from pytorch_lightning import Callback

from ahcore.utils.io import get_logger

logger = get_logger(__name__)


# Mockup
class TileWriter:
    def __init__(self, filename):
        logger.info(f"Created tilewriter for {filename}")

    def consume(self, generator):
        pass


def queue_generator(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        yield item


def writer_process(queue, filename, semaphore):
    writer = TileWriter(filename)
    writer.consume(queue_generator(queue))


class WriterCallback(Callback):
    def __init__(self, writer_class, max_concurrent_queues=5):
        self._writer_class = writer_class
        self._queues = {}
        self._processes = {}
        self._semaphore = Semaphore(max_concurrent_queues)

        self._gather_results = True

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        identifier = self._extract_identifier_from_batch(batch)
        logger.info(f"Got {identifier} for batch {batch_idx} on {trainer.global_rank}")

        # Check if a new queue and process should be created for this filename
        if identifier not in self._queues:
            logger.info(f"Got new {identifier} for batch {batch_idx} on {trainer.global_rank}")
            self._semaphore.acquire()
            logger.info(f"Acquired semaphore for new {identifier} for batch {batch_idx} on {trainer.global_rank}")
            self._queues[identifier] = Queue()
            process = Process(target=writer_process, args=(self._queues[identifier], identifier, self._semaphore))
            self._processes[identifier] = process
            process.start()

        # Add the batch to the queue for this filename
        self._queues[identifier].put(outputs)

        # Check if this is the last batch for this filename and signal the end if so
        # This is just in case
        if self._is_last_batch_for_file(identifier, batch_idx):
            self._queues[identifier].put(None)

    def on_inference_end(self, trainer, pl_module):
        # Wait for all processes to finish
        for process in self._processes.values():
            process.join()

    @staticmethod
    def _extract_identifier_from_batch(batch):
        return batch["filename"][0]

    def _is_last_batch_for_file(self, filename, batch_idx):
        # Implement logic to determine if this is the last batch for the given filename
        is_last_batch = True
        return is_last_batch
