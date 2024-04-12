from __future__ import annotations

import itertools
import json
import multiprocessing
import time
import warnings
from collections import namedtuple
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Generator, Optional, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback

from ahcore.callbacks import WriteFileCallback
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


class ComputeWsiMetricsCallback(Callback):
    def __init__(self, file_reader: FileImageReader, max_processes: int = 10, save_per_image: bool = True) -> None:
        """
        Callback to compute metrics on whole-slide images. This callback is used to compute metrics on whole-slide
        images in separate processes.

        Parameters
        ----------
        file_reader : FileImageReader
            The reader class to use to read the images, e.g H5FileImageReader or ZarrFileImageReader.
        max_processes : int
            The maximum number of concurrent processes.
        """
        self._data_description: Optional[DataDescription] = None
        self._file_reader: FileImageReader = file_reader
        self._max_processes: int = max_processes
        self._dump_dir: Optional[Path] = None
        self._save_per_image = save_per_image
        self._filenames: dict[Path, Path] = {}

        self._wsi_metrics: WSIMetricFactory | None = None
        self._class_names: dict[int, str] = {}
        self._data_manager = None
        self._validate_filenames_gen = None

        self._model_name: str | None = None

        self._validate_metadata_gen: Generator[ImageMetadata, None, None] | None = None

        self._dump_list: list[dict[str, str]] = []
        self._logger = get_logger(type(self).__name__)

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        if trainer.global_rank != 0:
            return

        if not isinstance(pl_module, AhCoreLightningModule):
            # TODO: Make a AhCoreCallback with these features
            raise ValueError("AhCoreLightningModule required for WriteTiffCallback.")

        self._model_name = pl_module.name

        _callback: Optional[WriteFileCallback] = None
        for idx, callback in enumerate(trainer.callbacks):  # type: ignore
            if isinstance(callback, WriteFileCallback):
                _callback = cast(WriteFileCallback, trainer.callbacks[idx])  # type: ignore
                break

        if _callback is None:
            raise ValueError(
                "WriteH5Callback is not in the trainer's callbacks. "
                "This is required before WSI metrics can be computed using this Callback"
            )

        self._dump_dir = _callback.dump_dir

        if pl_module.wsi_metrics is None:
            raise ValueError("WSI metrics are not set.")

        self._wsi_metrics = pl_module.wsi_metrics
        self._data_description = trainer.datamodule.data_description  # type: ignore

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

    def _create_validate_image_metadata_gen(
        self,
    ) -> Generator[ImageMetadata, None, None]:
        assert self._data_description
        assert self._data_manager
        gen = self._data_manager.get_image_metadata_by_split(
            manifest_name=self._data_description.manifest_name,
            split_version=self._data_description.split_version,
            split_category="validate",
        )
        for image_metadata in gen:
            yield image_metadata

    @property
    def _validate_metadata(self) -> Generator[ImageMetadata, None, None] | None:
        return self._validate_metadata_gen

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank != 0:
            return

        self._validate_metadata_gen = self._create_validate_image_metadata_gen()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.global_rank != 0:
            return

        if not self._dump_dir:
            raise ValueError("Dump directory is not set.")

    def compute_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> list[list[dict[str, dict[str, float]]]]:
        assert self._dump_dir
        assert self._data_description
        assert self._validate_metadata
        assert self._data_manager
        metrics = []

        with multiprocessing.Pool(processes=self._max_processes) as pool:
            results_to_filename: dict[list[dict[str, Any]], str] = {}
            completed_tasks = 0

            # Fill up the initial task pool
            for image_metadata in itertools.islice(self._validate_metadata, self._max_processes):
                logger.info("Metadata: %s", image_metadata)
                # Assemble the task data
                # filename", "h5_filename", "metadata", "mask", "annotations"
                task_data = prepare_task_data(
                    image_metadata.filename,
                    self._dump_dir,
                    pl_module,
                    self._data_description,
                    self._data_manager,
                )

                # Schedule task
                schedule_task(
                    task_data,
                    pool,
                    results_to_filename,
                    self._class_names,
                    self._data_description,
                    self._wsi_metrics,
                    self._save_per_image,
                )

            while results_to_filename:
                time.sleep(0.1)  # Reduce excessive polling
                # Check for completed tasks
                for result in list(results_to_filename.keys()):
                    if result.ready():
                        filename = results_to_filename.pop(result)
                        try:
                            metric = result.get()
                        except Exception as exc:
                            self._logger.error("%r generated an exception: %s" % (filename, exc))
                        else:
                            metrics.append(metric)
                            self._logger.debug("Metric for %r is %s" % (filename, metric))

                        completed_tasks += 1

                        # Schedule a new task if there are more filenames left in the generator
                        next_metadata = next(self._validate_metadata, None)
                        while next_metadata:
                            task_data = prepare_task_data(
                                next_metadata.filename,  # <-- Changed from image_metadata.filename
                                self._dump_dir,
                                pl_module,
                                self._data_description,
                                self._data_manager,
                            )

                            # Schedule task
                            schedule_task(
                                task_data,
                                pool,
                                results_to_filename,
                                self._class_names,
                                self._data_description,
                                self._wsi_metrics,
                                self._save_per_image,
                            )

                            next_metadata = next(self._validate_metadata, None)
        return metrics

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank != 0:
            return

        # Here we can check if everything is completed

        # if not self._dump_dir:
        #     raise ValueError("Dump directory is not set.")
        # if not self._wsi_metrics:
        #     raise ValueError("WSI metrics are not set.")
        # assert self._model_name  # This should be set in the setup()
        #
        # # Ensure that all cache files have been written
        # self._logger.debug("Computing metrics for %s predictions", len(self._filenames))
        # computed_metrics = self.compute_metrics(trainer, pl_module)
        # metrics = self._wsi_metrics.get_average_score(computed_metrics)
        # results_json_fn = (
        #     self._dump_dir / "outputs" / self._model_name / f"step_{pl_module.global_step}" / "results.json"
        # )
        # with open(results_json_fn, "w", encoding="utf-8") as json_file:
        #     json.dump(self._dump_list, json_file, indent=2)
        # self._wsi_metrics.reset()
        # # Reset stuff
        # self._dump_list = []
        # self._filenames = {}
        #
        # self._logger.debug("Metrics: %s", metrics)
        #
        # # TODO: Maybe put this elsewhere?
        # metrics = {f"validate/{k}": v for k, v in metrics.items()}
        # pl_module.log_dict(metrics, prog_bar=True)


TaskData = namedtuple("TaskData", ["filename", "h5_filename", "metadata", "mask", "annotations"])


def prepare_task_data(
    filename: Path,
    dump_dir: Path,
    pl_module: pl.LightningModule,
    data_description: DataDescription,
    data_manager: DataManager,
) -> TaskData:
    cache_filename = get_output_filename(
        dump_dir=dump_dir,
        input_path=data_description.data_dir / filename,
        model_name=str(pl_module.name),
        step=pl_module.global_step,
    )
    image = data_manager.get_image_by_filename(str(filename))
    metadata = fetch_image_metadata(image)
    mask, annotations = get_mask_and_annotations_from_record(data_description.annotations_dir, image)

    return TaskData(filename, cache_filename, metadata, mask, annotations)


def schedule_task(
    task_data: TaskData,
    pool: Pool,
    results_dict: dict[Any, str],  # Any because it will be a multiprocessing.pool.AsyncResult
    class_names: dict[int, str],
    data_description: DataDescription,
    wsi_metrics: WSIMetricFactory,
    save_per_image: bool,
) -> None:
    result = pool.apply_async(
        compute_metrics_for_case,
        args=(task_data, class_names, data_description, wsi_metrics, save_per_image),
    )
    results_dict[result] = task_data.filename


def compute_metrics_for_case(
    task_data: TaskData,
    class_names: dict[int, str],
    data_description: DataDescription,
    wsi_metrics: WSIMetricFactory,
    save_per_image: bool,
) -> list[dict[str, Any]]:
    # Extract the data from the namedtuple
    filename, h5_filename, metadata, mask, annotations = task_data
    dump_list = []

    logger.info("Computing metrics for %s", filename)

    with H5FileImageReader(h5_filename, stitching_mode=StitchingMode.CROP) as h5reader:
        dataset_of_validation_image = _ValidationDataset(
            data_description=data_description,
            native_mpp=metadata.mpp,
            mask=mask,
            annotations=annotations,
            reader=h5reader,
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
    if save_per_image:
        wsi_metrics_dictionary = {
            "image_fn": str(data_description.data_dir / metadata.filename),
            "uuid": filename.stem,
        }

        # TODO: These need to be removed, this is really weird.
        if filename.with_suffix(".tiff").is_file():
            wsi_metrics_dictionary["tiff_fn"] = str(filename.with_suffix(".tiff"))
        if filename.is_file():
            wsi_metrics_dictionary["h5_fn"] = str(filename)
        for metric in wsi_metrics._metrics:
            metric.get_wsi_score(str(filename))
            wsi_metrics_dictionary[metric.name] = {
                class_names[class_idx]: metric.wsis[str(filename)][class_idx][metric.name].item()
                for class_idx in range(data_description.num_classes)
            }
        dump_list.append(wsi_metrics_dictionary)

    return dump_list
