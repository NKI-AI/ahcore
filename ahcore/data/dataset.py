"""
Utilities to construct datasets and DataModule's from manifests.
"""

from __future__ import annotations

import bisect
import uuid as uuid_module
from typing import Any, Callable, Generator, Iterable, Iterator, Optional, Union, overload

import numpy as np
import pytorch_lightning as pl
import torch
from dlup.data.dataset import Dataset, TiledWsiDataset
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from ahcore.utils.data import DataDescription, basemodel_to_uuid
from ahcore.utils.io import fullname, get_cache_dir, get_logger
from ahcore.utils.manifest import DataManager, datasets_from_data_description
from ahcore.utils.types import DlupDatasetSample, _DlupDataset

logger = get_logger(__name__)


# TODO: This needs to be moved to dlup
class ConcatDataset(Dataset[DlupDatasetSample]):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Parameters
    ----------
    datasets : sequence
        List of datasets to be concatenated

    Notes
    -----
    Taken and adapted from pytorch 1.8.0 torch.utils.data.Dataset under BSD license.

    """

    datasets: list[Dataset[DlupDatasetSample]]
    cumulative_sizes: list[int]
    wsi_indices: dict[str, range]

    @staticmethod
    def cumsum(sequence: list[Dataset[DlupDatasetSample]]) -> list[int]:
        out_sequence, total = [], 0
        for item in sequence:
            length = len(item)
            out_sequence.append(length + total)
            total += length
        return out_sequence

    def __init__(self, datasets: Iterable[Dataset[DlupDatasetSample]]) -> None:
        super().__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, "datasets should not be an empty iterable"  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            if not hasattr(d, "__getitem__"):
                raise ValueError("ConcatDataset requires datasets to be indexable.")
        self.cumulative_sizes = self.cumsum(self.datasets)

    def index_to_dataset(self, idx: int) -> tuple[Dataset[DlupDatasetSample], int]:
        """Returns the dataset and the index of the sample in the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample in the concatenated dataset.

        Returns
        -------
        tuple[Dataset, int]
            Dataset and index of the sample in the dataset.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("Absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx], sample_idx

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    @overload
    def __getitem__(self, index: int) -> DlupDatasetSample:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[DlupDatasetSample]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> DlupDatasetSample | list[DlupDatasetSample]:
        """Returns the sample at the given index."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [{**self[i], "global_index": start + i} for i in range(start, stop, step or 1)]

        dataset, sample_idx = self.index_to_dataset(index)
        sample = dataset[sample_idx]
        sample["global_index"] = index
        return sample


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

        # Limit the number of samples to load for each stage, this is useful for debugging.
        self._limit_validate_samples = 2
        self._limit_fit_samples = None
        self._limit_predict_samples = None

    @property
    def data_manager(self) -> DataManager:
        return self._data_manager

    def setup(self, stage: str) -> None:
        if stage not in ("fit", "validate", "test", "predict"):
            raise ValueError(f"Stage should be one of fit, validate or test, found {stage}.")

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
        self, data_iterator: Iterator[_DlupDataset], batch_size: int, stage: str, distributed: bool = False
    ) -> Optional[DataLoader[DlupDatasetSample]]:
        if not data_iterator:
            return None

        limit_samples = getattr(self, f"_limit_{stage}_samples", None)

        def construct_dataset() -> ConcatDataset:
            datasets = []
            for idx, ds in enumerate(data_iterator):
                datasets.append(ds)

                if limit_samples and idx >= limit_samples:
                    break

            return ConcatDataset(datasets=datasets)

        self._logger.info("Constructing dataset for stage %s (this can take a while)", stage)
        dataset = self._load_from_cache(construct_dataset, stage=stage)
        setattr(self, f"{stage}_dataset", dataset)

        lengths = np.asarray([len(ds) for ds in dataset.datasets])
        self._logger.info(
            f"Dataset for stage {stage} has {len(dataset)} samples and the following statistics:\n"
            f" - Mean: {lengths.mean():.2f}\n"
            f" - Std: {lengths.std():.2f}\n"
            f" - Min: {lengths.min():.2f}\n"
            f" - Max: {lengths.max():.2f}"
        )

        sampler: Sampler[int]
        if stage == "fit":
            sampler = torch.utils.data.RandomSampler(data_source=dataset)

        elif stage == "predict" and distributed:
            # this is necessary because Lightning changes backend logic for predict
            # in particular, it will always return a non-repeating distributed sampler, causing deadlocks for callbacks
            sampler = DistributedSampler(
                dataset=dataset,
                shuffle=False,
            )

        else:
            sampler = torch.utils.data.SequentialSampler(data_source=dataset)

        return DataLoader(
            dataset,
            num_workers=self._num_workers,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True if stage == "fit" else False,
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
        setattr(self, "val_concat_dataset", val_dataloader.dataset if val_dataloader else None)
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
        distributed = self.trainer.world_size > 1 if self.trainer else False
        return self._construct_concatenated_dataloader(
            self._predict_data_iterator, batch_size=batch_size, stage="predict", distributed=distributed
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
