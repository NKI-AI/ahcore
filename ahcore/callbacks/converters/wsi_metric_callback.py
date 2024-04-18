from __future__ import annotations

import itertools
import time
import warnings
from collections import namedtuple
from multiprocessing import Process
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Generator, Optional, Type, cast

import pytorch_lightning as pl
import torch

from ahcore.callbacks import WriterCallback
from ahcore.callbacks.converters.common import ConvertCallbacks
from ahcore.lit_module import AhCoreLightningModule
from ahcore.metrics import WSIMetricFactory
from ahcore.readers import FileImageReader, StitchingMode
from ahcore.utils.callbacks import _ValidationDataset, get_output_filename
from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger
from ahcore.utils.manifest import DataManager, ImageMetadata, fetch_image_metadata, get_mask_and_annotations_from_record

logger = get_logger(__name__)


# Filter out a warning which is not relevant here
warnings.filterwarnings("ignore", message="It is recommended to use `sync_dist=True`*")


class ComputeWsiMetricsCallback(ConvertCallbacks):
    def __init__(
        self, reader_class: FileImageReader, max_concurrent_tasks: int = 1, save_per_image: bool = True
    ) -> None:
        """
        Callback to compute metrics on whole-slide images. This callback is used to compute metrics on whole-slide
        images in separate processes.

        Parameters
        ----------
        reader_class : FileImageReader
            The reader class to use to read the images, e.g H5FileImageReader or ZarrFileImageReader.
        max_processes : int
            The maximum number of concurrent processes.
        """
        super().__init__(max_concurrent_tasks=max_concurrent_tasks)
        self._data_description: Optional[DataDescription] = None
        self._file_reader: FileImageReader = reader_class
        self._max_processes: int = max_concurrent_tasks
        self._dump_dir: Optional[Path] = None
        self._save_per_image = save_per_image
        self._filenames: dict[Path, Path] = {}

        self._wsi_metrics: WSIMetricFactory | None = None
        self._class_names: dict[int, str] = {}
        self._data_manager = None

        self._model_name: str | None = None
        self._data_dir: Path

        self._storage = {}
        self.has_returns = True

    def setup(self, callback: WriterCallback, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if pl_module.wsi_metrics is None:
            raise ValueError("WSI metrics are not set.")

        self._wsi_metrics = pl_module.wsi_metrics
        self._data_description = trainer.datamodule.data_description  # type: ignore

        logger.info("Setting up WSI metrics callback")
        logger.info("WSI metrics: %s", self._wsi_metrics)
        logger.info("Data description: %s", self._data_description)
        logger.info("Data dir: %s", self._data_description.data_dir)

        # self._data_dir = self._data_description.data_dir

        # For mypy
        assert self._data_description
        index_map = self._data_description.index_map
        assert index_map

        if not self._data_description:
            raise ValueError("Data description is not set.")

        self._class_names = dict([(v, k) for k, v in index_map.items()])
        self._class_names[0] = "background"

        # Here we can query the database for the validation images
        self._data_manager: DataManager = trainer.datamodule.data_manager  # type: ignore
        # self._validate_metadata_gen = self._create_validate_image_metadata_gen()

        self._callback = callback
        self._trainer = trainer
        self._pl_module = pl_module
        self._stage = stage

        self._dump_dir = self._callback.dump_dir
        self._data_dir = self._pl_module.data_description.data_dir
        self._filenames_seen = []
        # MAYBE CAN CALL
        # Start worker processes
        for _ in range(self._max_concurrent_tasks):
            process = Process(target=self.worker)
            process.start()
            self._workers.append(process)

    def compute_metrics(self, filename, pl_module: pl.LightningModule):
        # So we have the filename of the image, but now we need to get it's metadata

        task_data = prepare_task_data(
            filename,
            self._dump_dir,
            self._data_dir,
            pl_module,
            self._data_description,
            self._data_manager,
        )
        logger.info("Computing metrics for %s", filename)
        logger.info("Task data: %s", task_data)

        curr_metrics = compute_metrics_for_case(
            task_data=task_data,
            image_reader=self._file_reader,
            class_names=self._class_names,
            data_description=self._data_description,
            wsi_metrics=self._wsi_metrics,
            save_per_image=self._save_per_image,
        )

        self._results_queue.put(curr_metrics)

        # TODO: Ajey, you can put the results in the Manager.Queue from base
        self._filenames_seen.append(filename)
        logger.info("Metrics: %s", curr_metrics)

    def process_task(self, filename: Path, cache_filename: Path) -> None:
        self.compute_metrics(filename, self._pl_module)


TaskData = namedtuple("TaskData", ["filename", "cache_filename", "metadata", "mask", "annotations"])


def prepare_task_data(
    filename: Path,
    dump_dir: Path,
    data_dir: Path,
    pl_module: pl.LightningModule,
    data_description: DataDescription,
    data_manager: DataManager,
) -> TaskData:
    cache_filename = get_output_filename(
        dump_dir=dump_dir,
        input_path=data_dir / filename,
        model_name=str(pl_module.name),
        step=pl_module.global_step,
    )

    image = data_manager.get_image_by_filename(str(filename.relative_to(data_dir)))
    metadata = fetch_image_metadata(image)
    mask, annotations = get_mask_and_annotations_from_record(data_description.annotations_dir, image)

    return TaskData(filename, cache_filename, metadata, mask, annotations)


def compute_metrics_for_case(
    task_data: TaskData,
    image_reader: Type[FileImageReader],
    class_names: dict[int, str],
    data_description: DataDescription,
    wsi_metrics: WSIMetricFactory,
    save_per_image: bool,
) -> list[dict[str, Any]]:
    # Extract the data from the namedtuple
    filename, cache_filename, metadata, mask, annotations = task_data

    logger.info("Computing metrics for %s", filename)

    with image_reader(cache_filename, stitching_mode=StitchingMode.CROP) as cache_reader:
        dataset_of_validation_image = _ValidationDataset(
            data_description=data_description,
            native_mpp=metadata.mpp,
            mask=mask,
            annotations=annotations,
            reader=cache_reader,
        )
        for sample in dataset_of_validation_image:
            prediction = torch.from_numpy(sample["prediction"]).unsqueeze(0).float()
            target = torch.from_numpy(sample["target"]).unsqueeze(0)
            roi = torch.from_numpy(sample["roi"]).unsqueeze(0) if sample["roi"] is not None else None

            wsi_metrics.process_batch(
                predictions=prediction,
                target=target,
                roi=roi,
                wsi_name=str(filename),
            )

    wsi_metrics_dictionary = {
        "image_fn": str(data_description.data_dir / metadata.filename),
        "uuid": filename.stem,
    }

    if filename.with_suffix(".tiff").is_file():
        wsi_metrics_dictionary["tiff_fn"] = str(filename.with_suffix(".tiff"))
    if filename.is_file():
        wsi_metrics_dictionary["cache_fn"] = str(filename)
    for metric in wsi_metrics._metrics:
        metric.get_wsi_score(str(filename))
        wsi_metrics_dictionary[metric.name] = {
            class_names[class_idx]: metric.wsis[str(filename)][class_idx][metric.name].item()
            for class_idx in range(data_description.num_classes)
        }

    return wsi_metrics_dictionary
