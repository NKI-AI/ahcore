from __future__ import annotations

import abc
from multiprocessing import Manager, Pool, Process, Queue
from pathlib import Path
from typing import Any, NamedTuple

import pytorch_lightning as pl

from ahcore.utils.callbacks import get_output_filename
from ahcore.utils.io import get_logger

logger = get_logger(__name__)


class CallbackOutput(NamedTuple):
    metrics: Any


class ConvertCallbacks(abc.ABC):
    def __init__(self, max_concurrent_tasks: int = 1):
        self._max_concurrent_tasks = max_concurrent_tasks
        self._pool = Pool(max_concurrent_tasks)
        self._task_queue: Queue = Queue()
        self._callback: Any
        self._trainer: pl.Trainer
        self._pl_module: pl.LightningModule
        self._stage: str
        self._dump_dir: Path
        self._workers: list[Process] = []

        self._manager = Manager()
        self._results_queue = self._manager.Queue()
        self._task_queue = Queue()

    def setup(self, callback: Any, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self._callback = callback
        self._trainer = trainer
        self._pl_module = pl_module
        self._stage = stage

        self._dump_dir = self._callback.dump_dir
        self._data_dir = self._pl_module.data_description.data_dir

        # Start worker processes
        for _ in range(self._max_concurrent_tasks):
            process = Process(target=self.worker)
            process.start()
            self._workers.append(process)

    @abc.abstractmethod
    def process_task(self, filename: Path, cache_filename: Path) -> Any:
        """Abstract method to process the task"""

    def schedule_task(self, filename: Path, cache_filename: Path):
        self._task_queue.put((filename, cache_filename))  # Put task into the queue for asynchronous processing

    def worker(self) -> None:
        """Worker function to process tasks from the queue and put results into results queue."""
        while True:
            task = self._task_queue.get()
            if task is None:  # Shutdown signal
                self._results_queue.put(None)  # Signal that this worker is done
                break
            filename, cache_filename = task
            logger.info("Processing task: %s %s", filename, cache_filename)

            result = self.process_task(filename, cache_filename)
            logger.info("Task completed: %s (from worker)", result)
            self._results_queue.put(result)  # Store the result

    @property
    def dump_dir(self) -> Path:
        return self._dump_dir

    def collect_results(self):
        logger.info("Collecting results")
        finished_workers = 0
        while finished_workers < self._max_concurrent_tasks:
            result = self._results_queue.get()
            logger.info("Result: %s", result)
            if result is None:
                finished_workers += 1
                logger.info(f"Worker completed, total finished: {finished_workers}")
                if finished_workers == self._max_concurrent_tasks:
                    logger.info("All workers have completed.")
                continue
            yield result

    def start(self, filename: str) -> None:
        cache_filename = get_output_filename(
            dump_dir=self._callback.dump_dir,
            input_path=self._pl_module.data_description.data_dir / filename,
            model_name=str(self._pl_module.name),
            step=self._pl_module.global_step,
        )

        # This should never not exist.
        assert Path(filename).exists()
        assert cache_filename.exists()

        self.schedule_task(filename=Path(filename), cache_filename=cache_filename)

    def shutdown_workers(self):
        logger.info("Shutting down workers...")
        for _ in range(self._max_concurrent_tasks):
            self._task_queue.put(None)  # Send shutdown signal
        for worker in self._workers:
            worker.join()  # Wait for all workers to finish
        logger.info("Workers shut down.")

