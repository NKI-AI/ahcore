"""
Module for the pre-transforms, which are the transforms that are applied before samples are outputted in a
dataset.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pyvips
import torch
from dlup.data.dataset import TileSample
from dlup.data.transforms import ConvertAnnotationsToMask, RenameLabels

from ahcore.exceptions import ConfigurationError
from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger
from ahcore.utils.types import DlupDatasetSample

PreTransformCallable = Callable[[Any], Any]

logger = get_logger(__name__)


class PreTransformTaskFactory:
    def __init__(self, transforms: list[PreTransformCallable]):
        """
        Pre-transforms are transforms that are applied to the samples directly originating from the dataset.
        These transforms are typically the same for the specific tasks (e.g., segmentation,
        detection or whole-slide classification).
        Each of these tasks has a specific constructor. In all cases, the final transforms convert the PIL image
        (as the image key of the output sample) to a tensor, and ensure that the sample dictionary can be collated.
        In ahcore, the augmentations are done separately and are part of the model in the forward function.

        Parameters
        ----------
        transforms : list
            List of transforms to be used.
        """
        # These are always finally added.
        transforms += [ImageToTensor(), AllowCollate(), SetTarget()]
        self._transforms = transforms

    @classmethod
    def for_segmentation(
        cls,
        data_description: DataDescription,
        requires_target: bool = True,
        multiclass: bool = True,
    ) -> PreTransformTaskFactory:
        """
        Pre-transforms for segmentation tasks. If the target is required these transforms are applied as follows:
        - Labels are renamed (for instance if you wish to map several labels to on specific class)
        - `Polygon` and `Point` annotations are converted to a mask
        - The mask is one-hot encoded.

        Parameters
        ----------
        data_description : DataDescription
        requires_target : bool
        multiclass : bool
            Set this to false if you have a categorical mask and you want to one-hot encode it.

        Returns
        -------
        PreTransformTaskFactory
            The `PreTransformTaskFactory` initialized for segmentation tasks.
        """
        transforms: list[PreTransformCallable] = []
        if not requires_target:
            return cls(transforms)

        if data_description.index_map is None:
            raise ConfigurationError("`index_map` is required for segmentation models when the target is required.")

        if data_description.remap_labels is not None:
            transforms.append(RenameLabels(remap_labels=data_description.remap_labels))

        transforms.append(
            ConvertAnnotationsToMask(
                roi_name=data_description.roi_name, index_map=data_description.index_map, multiclass=multiclass
            )
        )

        if not multiclass:
            transforms.append(OneHotEncodeMask(index_map=data_description.index_map))

        return cls(transforms)

    @classmethod
    def for_wsi_classification(
        cls, data_description: DataDescription, requires_target: bool = True
    ) -> PreTransformTaskFactory:
        transforms: list[PreTransformCallable] = []

        transforms.append(SampleNFeatures(n=1000))

        if not requires_target:
            return cls(transforms)

        index_map = data_description.index_map
        if index_map is None:
            raise ConfigurationError("`index_map` is required for classification models when the target is required.")

        label_keys = data_description.label_keys

        if label_keys is not None:
            transforms.append(SelectSpecificLabels(keys=label_keys))

        transforms.append(LabelToClassIndex(index_map=index_map))

        return cls(transforms)

    def __call__(self, data: DlupDatasetSample) -> DlupDatasetSample:
        for transform in self._transforms:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        return f"PreTransformTaskFactory(transforms={self._transforms})"


class SampleNFeatures:
    def __init__(self, n: int = 1000) -> None:
        self.n = n
        logger.warning(
            f"Sampling {n} features from the image. Sampling WITH replacement is done if there are not enough tiles."
        )

    def __call__(self, sample: DlupDatasetSample) -> DlupDatasetSample:
        features = sample["image"]

        # Get the dimensions of the image
        h = features.height  # Height
        w = features.width  # Width

        if h != 1:
            raise ValueError(f"Expected features to have a width dimension of 1, got {h}.")

        n_random_indices = (
            np.random.choice(w, self.n, replace=False) if w > self.n else np.random.choice(w, self.n, replace=True)
        )

        # Extract the selected columns (indices) from the image
        # Create a new image from the selected indices
        # todo: this can probably be done without a for-loop quicker
        selected_columns = [features.crop(idx, 0, 1, h) for idx in n_random_indices]

        # Combine the selected columns back into a single image
        sample["image"] = pyvips.Image.arrayjoin(selected_columns, across=1)

        return sample


class LabelToClassIndex:
    """
    Maps label values to class indices according to the index_map specified in the data description.
    Example:
        If there are two tasks:
            - Task1 with classes {A, B, C}
            - Task2 with classes {X, Y}
        Then an input sample could look like: {{"labels": {"Task1": "C", "Task2: "Y"}, ...}
        If the index map is: {"A": 0, "B": 1, "C": 2, "X": 0, "Y": 1}
        The returned sample will look like: {"labels": {"task1": 2, "task2": 1}, ...}
    """

    def __init__(self, index_map: dict[str, int]):
        self._index_map = index_map

    def __call__(self, sample: DlupDatasetSample) -> DlupDatasetSample:
        sample["labels"] = {
            label_name: self._index_map[label_value] for label_name, label_value in sample["labels"].items()
        }

        return sample


class SelectSpecificLabels:
    def __init__(self, keys: list[str] | str):
        if isinstance(keys, str):
            keys = [keys]
        self._keys = keys

    def __call__(self, sample: DlupDatasetSample) -> DlupDatasetSample:
        sample["labels"] = {
            label_key: label_value for label_key, label_value in sample["labels"].items() if label_key in self._keys
        }
        return sample


class OneHotEncodeMask:
    def __init__(self, index_map: dict[str, int]):
        """Create the one-hot encoding of the mask for segmentation.
        If we have `N` classes, the result will be an `(B, N + 1, H, W)` tensor, where the first sample is the
        background.
        Parameters
        ----------
        index_map : dict[str, int]
            Index map mapping the label name to the integer value it has in the mask.
        """
        self._index_map = index_map

        # Check the max value in the mask
        self._largest_index = max(index_map.values())

    def __call__(self, sample: DlupDatasetSample) -> DlupDatasetSample:
        mask = sample["annotation_data"]["mask"]

        new_mask = np.zeros((self._largest_index + 1, *mask.shape))
        for idx in range(self._largest_index + 1):
            new_mask[idx] = (mask == idx).astype(np.float32)

        sample["annotation_data"]["mask"] = new_mask
        return sample


def one_hot_encoding(index_map: dict[str, int], mask: npt.NDArray[np.int_ | np.float_]) -> npt.NDArray[np.float32]:
    """
    functional interface to convert labels/predictions into one-hot codes

    Parameters
    ----------
    index_map : dict[str, int]
        Index map mapping the label name to the integer value it has in the mask.

    mask: npt.NDArray
        The numpy array of model predictions or ground truth labels.

    Returns
    -------
    new_mask: npt.NDArray
        One-hot encoded output
    """
    largest_index = max(index_map.values())
    new_mask = np.zeros((largest_index + 1, *mask.shape), dtype=np.float32)
    for idx in range(largest_index + 1):
        new_mask[idx] = mask == idx
    return new_mask


class AllowCollate:
    """Path objects cannot be collated in the standard pytorch collate function.
    This transform converts the path to a string. Same holds for the annotations and labels
    """

    def __call__(self, sample: TileSample) -> dict[str, Any]:
        # Path objects cannot be collated
        output = dict(sample.copy())
        for key in sample:
            if key == "path":
                output["path"] = str(sample["path"])
            if key in ["annotation_data", "annotations"]:
                # remove annotation_data and annotations keys from output
                del output[key]
            if key == "labels" and sample["labels"] is None:
                del output[key]

        return output


class SetTarget:
    def __call__(self, sample: DlupDatasetSample) -> DlupDatasetSample:
        if "annotations_data" in sample and "mask" in sample["annotation_data"] and "labels" in sample.keys():
            sample["target"] = (sample["annotation_data"]["mask"], sample["labels"])
        elif "annotations_data" in sample and "mask" in sample["annotation_data"]:
            sample["target"] = sample["annotation_data"]["mask"]
        elif "labels" in sample.keys():
            if len(sample["labels"].keys()) == 1:
                # if there is only one label, then we just set this without retaining the key
                # this makes it compatible with standard loss functions
                sample["labels"] = next(iter(sample["labels"].values()))
            sample["target"] = sample["labels"]
        else:
            logging.warning("No target set")

        return sample


class ImageToTensor:
    """
    Transform to translate the output of a dlup dataset to data_description supported by AhCore
    """

    def __call__(self, sample: DlupDatasetSample) -> dict[str, DlupDatasetSample]:
        tile: pyvips.Image = sample["image"]
        # Flatten the image to remove the alpha channel, using white as the background color
        using_features = False

        if tile.bands > 4:
            # assuming that more than four bands/channels means that we are handling features
            using_features = True
            tile_ = tile
        else:
            tile_ = tile.flatten(background=[255, 255, 255])  # todo: check if this doesn't mess up features

        # Convert VIPS image to a numpy array then to a torch tensor
        np_image = tile_.numpy()
        if using_features:
            # n_tiles x 1 x feature_dim --> n_tiles x feature_dim
            sample["image"] = torch.from_numpy(np_image).squeeze(1).float()
        else:
            # h x w x c --> c x h x w
            sample["image"] = torch.from_numpy(np_image).permute(2, 0, 1).float()

        if sample["image"].sum() == 0:
            raise RuntimeError(f"Empty tile for {sample['path']} at {sample['coordinates']}")

        if "labels" in sample:
            for key, value in sample["labels"].items():
                sample["labels"][key] = torch.tensor(value)

        # annotation_data is added by the ConvertPolygonToMask transform.
        if "annotation_data" not in sample:
            return sample

        if "mask" in sample["annotation_data"]:
            mask = sample["annotation_data"]["mask"]
            if len(mask.shape) == 2:
                # Mask is not one-hot encoded
                mask = mask[np.newaxis, ...]
            sample["annotation_data"]["mask"] = torch.from_numpy(mask).float()

        if "roi" in sample["annotation_data"] and sample["annotation_data"]["roi"] is not None:
            roi = sample["annotation_data"]["roi"]
            sample["roi"] = torch.from_numpy(roi[np.newaxis, ...]).float()

        sample["mpp"] = torch.tensor(
            sample["mpp"], dtype=torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        return sample

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
