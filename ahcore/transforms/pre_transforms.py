"""
Module for the pre-transforms, which are the transforms that are applied before samples are outputted in a
dataset.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt
import torch
from dlup.data.dataset import TileSample
from dlup.data.transforms import ContainsPolygonToLabel, ConvertAnnotationsToMask, RenameLabels
from torchvision.transforms import functional as F

from ahcore.exceptions import ConfigurationError
from ahcore.transforms.dlup_transforms import ConvertPointAnnotationsToMask
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
        transforms += [
            ImageToTensor(),
            AllowCollate(),
        ]
        self._transforms = transforms

    @classmethod
    def for_detection(cls, data_description: DataDescription, requires_target: bool = True) -> PreTransformTaskFactory:
        """
        Pretransforms for detection tasks. If the target is required these transforms are applied as follows:
        - Labels are renamed (for instance if you wish to map several labels to on specific class)
        - `Polygon` and `Point` annotations are converted to a mask
        - The mask is one-hot encoded.
        Parameters
        ----------
        data_description : DataDescription
        requires_target : bool
        Returns
        -------
        PreTransformTaskFactory
            The `PreTransformTaskFactory` initialized for detection tasks.
        """
        transforms: list[PreTransformCallable] = []
        if not requires_target:
            return cls(transforms)

        if data_description.index_map is None:
            raise ConfigurationError("`index_map` is required for segmentation models when the target is required.")

        if data_description.remap_labels is not None:
            transforms.append(RenameLabels(remap_labels=data_description.remap_labels))

        if data_description.point_radius_microns is None or data_description.training_grid.mpp is None:
            raise ConfigurationError(
                "`points_radius_micro_m` and `training_grid.mpp` need to be defined for calculation of radius."
            )
        else:
            _radius_pixels = int(round(data_description.point_radius_microns / data_description.training_grid.mpp))
            if _radius_pixels <= 0:
                raise ConfigurationError(
                    f"Point conversion radius in pixels must be postive but got {_radius_pixels=}."
                )

        transforms.extend(
            [
                ConvertPointAnnotationsToMask(
                    roi_name=data_description.roi_name, index_map=data_description.index_map, radius=_radius_pixels
                ),
                OneHotEncodeMask(index_map=data_description.index_map),
                AnnotationsToTensor(index_map=data_description.index_map, ann_type="points"),
            ]
        )

        return cls(transforms)

    @classmethod
    def for_segmentation(
        cls, data_description: DataDescription, requires_target: bool = True
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
            ConvertAnnotationsToMask(roi_name=data_description.roi_name, index_map=data_description.index_map)
        )
        transforms.append(OneHotEncodeMask(index_map=data_description.index_map))

        return cls(transforms)

    @classmethod
    def for_wsi_classification(
        cls, data_description: DataDescription, requires_target: bool = True
    ) -> PreTransformTaskFactory:
        transforms: list[PreTransformCallable] = []
        if not requires_target:
            return cls(transforms)

        index_map = data_description.index_map
        if index_map is None:
            raise ConfigurationError("`index_map` is required for classification models when the target is required.")

        transforms.append(LabelToClassIndex(index_map=index_map))

        return cls(transforms)

    @classmethod
    def for_tile_classification(cls, roi_name: str, label: str, threshold: float) -> PreTransformTaskFactory:
        """Tile classification is based on a transform which checks if a polygon is present for a given threshold"""
        convert_annotations = ContainsPolygonToLabel(roi_name=roi_name, label=label, threshold=threshold)
        return cls([convert_annotations])

    def __call__(self, data: DlupDatasetSample) -> DlupDatasetSample:
        for transform in self._transforms:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        return f"PreTransformTaskFactory(transforms={self._transforms})"


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


class ImageToTensor:
    """
    Transform to translate the output of a dlup dataset to data_description supported by AhCore
    """

    def __call__(self, sample: DlupDatasetSample) -> dict[str, DlupDatasetSample]:
        sample["image"] = F.pil_to_tensor(sample["image"].convert("RGB")).float()

        if sample["image"].sum() == 0:
            raise RuntimeError(f"Empty tile for {sample['path']} at {sample['coordinates']}")

        # annotation_data is added by the ConvertPolygonToMask transform.
        if "annotation_data" not in sample:
            return sample

        if "mask" in sample["annotation_data"]:
            mask = sample["annotation_data"]["mask"]
            if len(mask.shape) == 2:
                # Mask is not one-hot encoded
                mask = mask[np.newaxis, ...]
            sample["target"] = torch.from_numpy(mask).float()

        if "roi" in sample["annotation_data"] and sample["annotation_data"]["roi"] is not None:
            roi = sample["annotation_data"]["roi"]
            sample["roi"] = torch.from_numpy(roi[np.newaxis, ...]).float()

        sample["mpp"] = torch.tensor(
            sample["mpp"], dtype=torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        return sample

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class AnnotationsToTensor:
    def __init__(self, index_map: dict[str, int], ann_type: Literal["points", "boxes"] = "points") -> None:
        """Transform detection annotations to Tensor object.
        If we have `K` classes, the output Tensor will have shape (N, 3) or (N, 5), where `N = \\sum_{k=0}^K N_k`
        with `N_k` as the number of annotations of class `C_k`. When `ann_type` is 'points' each sample will have
        (x, y, label_index). When `ann_type` is 'boxes' each sample will have (x, y, width, heigth label_index).

        Parameters
        ----------
        index_map: dict[str, int]
            Index map mapping the label name to the integer value it has in the mask.
        ann_type: str
            Annotation type to convert to tensor. Can be either 'points' or 'boxes'
        """
        if ann_type not in ["points", "boxes"]:
            raise ConfigurationError(f"`ann_type` must be 'points' or 'boxes' but got {ann_type=}")
        self._index_map = index_map
        self._ann_type = ann_type
        self._output_size = 3 if self._ann_type == "points" else 5

    def __call__(self, sample: DlupDatasetSample) -> DlupDatasetSample:
        if "annotation_data" not in sample:
            return sample

        if self._ann_type not in sample["annotation_data"]:
            sample[self._ann_type] = torch.empty((0, self._output_size))
            return sample

        annotations: dict[str, Any] = sample["annotation_data"][self._ann_type]
        _annotations = []
        for label, index in self._index_map.items():
            if label not in annotations:
                continue

            label_annotations = [
                [*ann, index] if self._ann_type == "points" else [*itertools.chain.from_iterable(ann), index]
                for ann in annotations[label]
            ]

            _annotations.extend(label_annotations)

        sample[self._ann_type] = (
            torch.from_numpy(np.asarray(_annotations)) if _annotations else torch.empty((0, self._output_size))
        )
        return sample

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
