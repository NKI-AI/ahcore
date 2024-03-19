"""
Utilities to construct datasets and DataModule's from manifests.
"""
from __future__ import annotations

import uuid as uuid_module
from typing import Any, Callable, Generator, Iterator, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from dlup.data.dataset import ConcatDataset, TiledWsiDataset
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader, Sampler

import ahcore.data.samplers
from ahcore.utils.data import DataDescription, basemodel_to_uuid, collate_fn_annotations
from ahcore.utils.io import fullname, get_cache_dir, get_logger
from ahcore.utils.manifest import DataManager, datasets_from_data_description
from ahcore.utils.types import DlupDatasetSample, _DlupDataset


class DlupDataModule(pl.LightningDataModule):
    """Datamodule for the Ahcore framework. This datamodule is based on `dlup`."""

    def __init__(
        self,
        data_description: DataDescription,
        pre_transform: Callable[[bool], Callable[[DlupDatasetSample], DlupDatasetSample]],
        batch_size: int = 32,  # noqa,pylint: disable=unused-argument
        validate_batch_size: int | None = None,  # noqa,pylint: disable=unused-argument
        num_workers: int = 16,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """
        Construct a DataModule based on a manifest.

        Parameters
        ----------
        data_description : DataDescription
            See `ahcore.utils.data.DataDescription` for more information.
        pre_transform : Callable
            A pre-transform is a callable which is directly applied to the output of the dataset before collation in
            the dataloader. The transforms typically convert the image in the output to a tensor, convert the
            `WsiAnnotations` to a mask or similar.
        batch_size : int
            The batch size of the data loader.
        validate_batch_size : int, optional
            Sometimes the batch size for validation can be larger. If so, set this variable. Will also use this for
            prediction.
        num_workers : int
            The number of workers used to fetch tiles.
        persistent_workers : bool
            Whether to use persistent workers. Check the pytorch documentation for more information.
        pin_memory : bool
            Whether to use cuda pin workers. Check the pytorch documentation for more information.
        """
        super().__init__()
        self._logger = get_logger(fullname(self))

        self.save_hyperparameters(
            logger=True,
            ignore=[
                "data_description",
                "pre_transform",
                "data_dir",
                "annotations_dir",
                "num_workers",
                "persistent_workers",
                "pin_memory",
            ],
        )  # save all relevant hyperparams

        # Data settings
        self.data_description: DataDescription = data_description

        self._data_manager = DataManager(database_uri=data_description.manifest_database_uri)

        self._batch_size = self.hparams.batch_size  # type: ignore
        self._validate_batch_size = self.hparams.validate_batch_size  # type: ignore

        mask_threshold = data_description.mask_threshold
        if mask_threshold is None:
            mask_threshold = 0.0
        self._mask_threshold = mask_threshold

        self._pre_transform = pre_transform

        # DataLoader settings
        collate_fn = (
            collate_fn_annotations if self.data_description.use_points or self.data_description.use_boxes else None
        )
        self._collate_fn = collate_fn
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

        self._fit_data_iterator: Iterator[_DlupDataset] | None = None
        self._validate_data_iterator: Iterator[_DlupDataset] | None = None
        self._test_data_iterator: Iterator[_DlupDataset] | None = None
        self._predict_data_iterator: Iterator[_DlupDataset] | None = None

        # Variables to keep track if a dataset has already be constructed (it's a slow operation)
        self._already_called: dict[str, bool] = {
            "fit": False,
            "validate": False,
            "test": False,
            "predict": False,
        }
        self._num_classes = data_description.num_classes

    @property
    def data_manager(self) -> DataManager:
        return self._data_manager

    def setup(self, stage: str) -> None:
        if stage not in (e.value for e in TrainerFn):
            raise ValueError(f"Stage should be one of {TrainerFn}")

        if stage and self._already_called[stage]:
            return

        self._logger.info("Constructing dataset iterator for stage %s", stage)

        def dataset_iterator() -> Generator[TiledWsiDataset, None, None]:
            gen = datasets_from_data_description(
                db_manager=self._data_manager,
                data_description=self.data_description,
                transform=self._pre_transform(
                    requires_target=True if stage != "predict" else False  # type: ignore
                ),  # This bool adds the target
                stage=stage,
            )
            for dataset in gen:
                yield dataset

        setattr(self, f"_{stage}_data_iterator", dataset_iterator())

    def _construct_concatenated_dataloader(
        self, data_iterator: Iterator[_DlupDataset], batch_size: int, stage: str
    ) -> Optional[DataLoader[DlupDatasetSample]]:
        if not data_iterator:
            return None

        def _construct_dataset_weights(dataset: _DlupDataset) -> torch.Tensor:
            def _get_sample_weights(sample: DlupDatasetSample) -> torch.Tensor:
                sample_target: Optional[torch.Tensor] = sample.get("target", None)
                if sample_target is None:
                    raise ValueError("Cannot convert None target to distribution.")
                _, width, height = sample_target.shape
                total_pixels = width * height
                sample_weights = sample_target.sum(dim=(1, 2)) / total_pixels
                return sample_weights

            # Annotations_only only works on specific branch. Otherwise image will get retrieved as well
            dataset.annotations_only = True  # type: ignore
            dataset_weight_matrix = torch.stack([_get_sample_weights(sample) for sample in dataset])
            dataset.annotations_only = False  # type: ignore
            return dataset_weight_matrix

        def construct_dataset() -> tuple[ConcatDataset[DlupDatasetSample], torch.Tensor | None]:
            datasets = []
            weights: list = []
            class_weights = (
                torch.Tensor(self.data_description.class_weights)
                if self.data_description.class_weights is not None
                else None
            )
            for ds in data_iterator:
                datasets.append(ds)
                if stage == "fit" and class_weights is not None:
                    weight_matrix = _construct_dataset_weights(ds)
                    sample_weights = (weight_matrix * class_weights).sum(dim=1)
                    weights.append(sample_weights)

            return ConcatDataset(datasets), (
                torch.cat(weights) if stage == "fit" and class_weights is not None else None
            )

        self._logger.info("Constructing dataset for stage %s (this can take a while)", stage)
        dataset, weights = self._load_from_cache(construct_dataset, stage=stage)
        setattr(self, f"{stage}_dataset", dataset)

        lengths = np.asarray([len(ds) for ds in dataset.datasets])
        self._logger.info(
            f"Dataset for stage {stage} has {len(dataset)} samples and the following statistics:\n"
            f" - Mean: {lengths.mean():.2f}\n"
            f" - Std: {lengths.std():.2f}\n"
            f" - Min: {lengths.min():.2f}\n"
            f" - Max: {lengths.max():.2f}"
        )

        batch_sampler: Sampler[list[int]]
        if stage == "fit":
            if weights is not None:
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(dataset), replacement=True
                )
            else:
                sampler = torch.utils.data.RandomSampler(data_source=dataset, replacement=True)  # type: ignore
            batch_sampler = torch.utils.data.BatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=True,
            )

        elif stage == "predict":
            batch_sampler = ahcore.data.samplers.WsiBatchSamplerPredict(
                dataset=dataset,
                batch_size=batch_size,
            )

        else:
            batch_sampler = ahcore.data.samplers.WsiBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
            )

        return DataLoader(
            dataset,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            batch_sampler=batch_sampler,
            persistent_workers=self._persistent_workers,
            pin_memory=self._pin_memory,
        )

    def _load_from_cache(self, func: Callable[[], Any], stage: str, *args: Any, **kwargs: Any) -> Any:
        name = fullname(func)
        path = get_cache_dir() / stage / name
        filename = path / f"{self.uuid}.pkl"
        if not filename.is_file():
            path.mkdir(exist_ok=True, parents=True)
            self._logger.info("Caching %s", name)

            obj = func(*args, **kwargs)

            with open(filename, "wb") as file:
                torch.save(obj, file)
        else:
            with open(filename, "rb") as file:
                self._logger.info("Loading %s from cache %s file", name, filename)
                obj = torch.load(file)

        return obj

    def train_dataloader(self) -> Optional[DataLoader[DlupDatasetSample]]:
        if not self._fit_data_iterator:
            self.setup("fit")
        assert self._fit_data_iterator
        return self._construct_concatenated_dataloader(
            self._fit_data_iterator,
            batch_size=self._batch_size,
            stage="fit",
        )

    def val_dataloader(self) -> Optional[DataLoader[DlupDatasetSample]]:
        if not self._validate_data_iterator:
            self.setup("validate")

        batch_size = self._validate_batch_size if self._validate_batch_size else self._batch_size
        assert self._validate_data_iterator
        val_dataloader = self._construct_concatenated_dataloader(
            self._validate_data_iterator,
            batch_size=batch_size,
            stage="validate",
        )
        if val_dataloader:
            setattr(self, "val_concat_dataset", val_dataloader.dataset)
        else:
            setattr(self, "val_concat_dataset", None)
        return val_dataloader

    def test_dataloader(self) -> Optional[DataLoader[DlupDatasetSample]]:
        if not self._test_data_iterator:
            self.setup("test")
        batch_size = self._validate_batch_size if self._validate_batch_size else self._batch_size
        assert self._validate_data_iterator
        return self._construct_concatenated_dataloader(
            self._validate_data_iterator, batch_size=batch_size, stage="test"
        )

    def predict_dataloader(self) -> Optional[DataLoader[DlupDatasetSample]]:
        if not self._predict_data_iterator:
            self.setup("predict")
        batch_size = self._validate_batch_size if self._validate_batch_size else self._batch_size
        assert self._predict_data_iterator
        return self._construct_concatenated_dataloader(
            self._predict_data_iterator, batch_size=batch_size, stage="predict"
        )

    def teardown(self, stage: str | None = None) -> None:
        if stage is not None:
            getattr(self, f"_{stage}_data_iterator").__del__()
        self._data_manager.close()

    @property
    def uuid(self) -> uuid_module.UUID:
        """This property is used to create a unique cache file for each dataset. The constructor of this dataset
        is completely determined by the data description, including the pre_transforms. Therefore, we can use the
        data description to create an uuid that is unique for each datamodule.

        The uuid is computed by hashing the data description using the `basemodel_to_uuid` function, which uses
        a sha256 hash of the pickled object and converts it to an UUID.
        As pickles can change with python versions, this uuid will be different when using different python versions.

        Returns
        -------
        str
            A unique identifier for this datamodule.
        """
        return basemodel_to_uuid(self.data_description)
