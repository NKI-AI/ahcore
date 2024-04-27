from __future__ import annotations

import abc
from multiprocessing import Lock, Manager, Pool, Process, Queue, Value
from pathlib import Path
from typing import Any, Generator, NamedTuple

import pytorch_lightning as pl

from ahcore.lit_module import AhCoreLightningModule
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
        self._pl_module: AhCoreLightningModule
        self._stage: str
        self._workers: list[Process] = []

        # Paths
        self._dump_dir: Path
        self._data_dir: Path

        self._manager = Manager()
        self._results_queue = self._manager.Queue()
        self._task_queue = Queue()

        self._completed_tasks = Value("i", 0)  # 'i' Tracks completed tasks
        self._completed_tasks_lock = Lock()  # To ensure thread-safe increments

        self.has_returns: bool

    def setup(self, callback: Any, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self._callback = callback
        self._trainer = trainer
        self._pl_module = pl_module
        self._stage = stage

        self._dump_dir = self._callback.dump_dir
        self._data_dir = self._pl_module.data_description.data_dir

    @abc.abstractmethod
    def process_task(self, filename: Path, cache_filename: Path) -> Any:
        """Abstract method to process the task"""

    def schedule_task(self, filename: Path, cache_filename: Path) -> None:
        """Schedule a task for processing"""
        self._task_queue.put((filename, cache_filename))  # Put task into the queue for asynchronous processing

    def worker(self) -> None:
        """Worker function to process tasks from the queue and put results into results queue."""
        logger.info("Starting worker for %s", type(self).__name__)
        while True:
            task = self._task_queue.get()
            if task is None:  # Shutdown signal
                self._results_queue.put(None)  # Signal that this worker is done
                break
            filename, cache_filename = task
            logger.info(
                "Processing task: %s (this is %s)",
                filename,
                type(self).__name__,
            )

            result = self.process_task(filename, cache_filename)
            logger.info("Task completed: %s (from %s)", result, type(self).__name__)
            self._results_queue.put(result)  # Store the result
            logger.info("Completion counter before increment: %s", self.completed_tasks)
            with self._completed_tasks_lock:
                self._completed_tasks.value += 1
                logger.info("Completion counter after increment: %s", self.completed_tasks)
            logger.info("Completion counter after increment: %s", self.completed_tasks)

    def reset_counters(self):
        with self._completed_tasks_lock:
            self._completed_tasks.value = 0
            logger.info("Task counters reset.")

    @property
    def completed_tasks(self) -> int:
        return self._completed_tasks.value

    @property
    def dump_dir(self) -> Path:
        return self._dump_dir

    def collect_results(self) -> Generator:
        finished_workers = 0

        logger.info("Collecting results %s %s", finished_workers, self._max_concurrent_tasks)
        while finished_workers < self._max_concurrent_tasks:
            result = self._results_queue.get()
            logger.info("Result: %s", result)
            if result is None:
                finished_workers += 1
                if finished_workers == self._max_concurrent_tasks:
                    logger.info("All workers have completed.")
                continue
            yield result

    def start(self, filename: str) -> None:
        # Need to use training step here or otherwise the next few callbacks will not work
        logger.info("Starting %s for %s", type(self).__name__, filename)
        cache_filename = get_output_filename(
            dump_dir=self._callback.dump_dir,
            input_path=self._data_dir / filename,
            model_name=str(self._pl_module.name),
            counter=f"{self._pl_module.current_epoch}_{self._pl_module.validation_counter}",
        )

        # This should never not exist.
        assert Path(filename).exists()
        assert cache_filename.exists()

        self.schedule_task(filename=Path(filename), cache_filename=cache_filename)

    def start_workers(self):
        for _ in range(self._max_concurrent_tasks):
            process = Process(target=self.worker)
            process.start()
            self._workers.append(process)
        logger.info("Workers started.")

    def shutdown_workers(self) -> None:
        logger.info("Shutting down workers...")
        for _ in range(self._max_concurrent_tasks):
            self._task_queue.put(None)  # Send shutdown signal
        for worker in self._workers:
            worker.join()  # Wait for all workers to finish
        logger.info("Workers shut down.")
