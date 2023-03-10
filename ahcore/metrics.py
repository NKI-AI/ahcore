# encoding: utf-8
"""Metrics module, including factory.
"""
from __future__ import annotations

import abc
from typing import List, Tuple

import torch
import torch.nn.functional as F  # noqa

from ahcore.exceptions import ConfigurationError
from ahcore.utils.data import DataDescription


class AhCoreMetric:
    def __init__(self, data_description: DataDescription) -> None:
        """Initialize the metric class"""
        self._data_description = data_description
        self.name: str | None = None

    @abc.abstractmethod
    def __call__(
        self, predictions: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Call metric computation"""


class DiceMetric(AhCoreMetric):
    def __init__(self, data_description: DataDescription) -> None:
        """
        Metric computing dice over classes. The classes are derived from the index_map that's defined in the
        data_description.

        First, a softmax is taken over the predictions, followed by a softmax. Then, if there is a ROI available, the
        input and target are masked with this ROI. This is followed by an argmax over the predictions and target,
        resulting in a tensor of shape (batch_size, height, width) with values in [0, num_classes - 1]. The dice is then
        computed over each class.

        We use as definition for the dice score:
        :math:`\text{dice} = 2 * \frac{|X| \intersection |Y|}{|X| + |Y|}` where :math:`|X|` is the number of voxels in
        the prediction, :math:`|Y|` is the number of voxels in the target, and :math:`\intersection` is the intersection
        of :math:`X` and :math:`Y`.

        The `__call__` returns the dice score for each class, with the class name (prefixed with dice/) as key
        in a dictionary.

        Parameters
        ----------
        data_description : DataDescription
        """
        super().__init__(data_description=data_description)
        self._num_classes = self._data_description.num_classes

        # Invert the index map
        _index_map = {}
        if self._data_description.index_map is None:
            raise ConfigurationError("`index_map` is required for to setup the dice metric.")
        else:
            _index_map = self._data_description.index_map

        _label_to_class = {v: k for k, v in _index_map.items()}
        _label_to_class[0] = "background"
        self._label_to_class = _label_to_class

        self.name = "dice"

    def __call__(self, predictions, target, roi: torch.Tensor | None):
        dice_components = _get_intersection_and_cardinality(predictions, target, roi, self._num_classes)
        dices = []
        for intersection, cardinality in dice_components:
            # dice_score is a float
            dice_score = _compute_dice(intersection, cardinality)
            dices.append(dice_score)

        output = {f"{self.name}/{self._label_to_class[idx]}": dices[idx] for idx in range(0, self._num_classes)}
        return output

    def __repr__(self):
        return f"{type(self).__name__}(num_classes={self._num_classes})"


class MetricFactory:
    """Factory to create the metrics. These are fixed for the different tasks (e.g., segmentation, detection, whole-slide-level
    classification.
    """

    def __init__(self, metrics: list[AhCoreMetric]) -> None:
        """
        Parameters
        ----------
        metrics : list
            List of metrics of type `AhCoreMetric`.
        """
        super().__init__()
        names = [metric.name for metric in metrics]
        if len(set(names)) != len(names):
            raise RuntimeError("Each individual metric must have a different name.")

        self._metrics = metrics

    @classmethod
    def for_segmentation(cls, *args, **kwargs):
        dices = DiceMetric(*args, **kwargs)
        return cls([dices])

    @classmethod
    def for_wsi_classification(cls, config):
        raise NotImplementedError

    @classmethod
    def for_tile_classification(cls, roi_name: str, label: str, threshold: float):
        raise NotImplementedError

    def __call__(
        self, predictions: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        output = {}
        for metric in self._metrics:
            output.update(metric(predictions, target, roi=roi))
        return output

    def __repr__(self):
        return f"{type(self).__name__}(metrics={self._metrics})"


class WSIMetric(abc.ABC):
    def __init__(self, data_description: DataDescription) -> None:
        """Initialize the WSI metric class"""
        self._data_description = data_description
        self.name: str | None = None

    @abc.abstractmethod
    def process_batch(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def get_wsi_score(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def get_average_score(self, *args, **kwargs) -> dict[str, float]:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class WSIDiceMetric(WSIMetric):
    """WSI Dice metric class, computes the dice score over the whole WSI"""

    def __init__(self, data_description: DataDescription):
        super().__init__(data_description=data_description)
        self.wsis = {}
        self._num_classes = self._data_description.num_classes

        # Invert the index map
        _index_map = {}
        if self._data_description.index_map is None:
            raise ConfigurationError("`index_map` is required for to setup the wsi-dice metric.")
        else:
            _index_map = self._data_description.index_map

        _label_to_class = {v: k for k, v in _index_map.items()}
        _label_to_class[0] = "background"
        self._label_to_class = _label_to_class

        self.name = "wsi_dice"

    def process_batch(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor | None,
        wsi_name: str,
    ) -> None:
        if wsi_name not in self.wsis:
            self._initialize_wsi_dict(wsi_name)
        dice_components = _get_intersection_and_cardinality(predictions, target, roi, self._num_classes)
        for class_idx, (intersection, cardinality) in enumerate(dice_components):
            self.wsis[wsi_name][class_idx]["intersection"] += intersection
            self.wsis[wsi_name][class_idx]["cardinality"] += cardinality

    def get_wsi_score(self, wsi_name: str) -> None:
        for class_idx in self.wsis[wsi_name]:
            intersection = self.wsis[wsi_name][class_idx]["intersection"]
            cardinality = self.wsis[wsi_name][class_idx]["cardinality"]
            self.wsis[wsi_name][class_idx]["dice"] = _compute_dice(intersection, cardinality)

    def _initialize_wsi_dict(self, wsi_name: str) -> None:
        self.wsis[wsi_name] = {
            class_idx: {"intersection": 0, "cardinality": 0, "dice": None} for class_idx in range(self._num_classes)
        }

    def get_average_score(self) -> dict[str, float]:
        dices = {class_idx: [] for class_idx in range(self._num_classes)}
        for wsi_name in self.wsis:
            self.get_wsi_score(wsi_name)
            for class_idx in range(self._num_classes):
                dices[class_idx].append(self.wsis[wsi_name][class_idx]["dice"].item())
        avg_dict = {
            f"{self.name}/{self._label_to_class[idx]}": sum(value) / len(value) for idx, value in dices.items()
        }
        return avg_dict

    def reset(self):
        self.wsis = {}

    def __repr__(self):
        return f"{type(self).__name__}(num_classes={self._num_classes})"


class WSIMetricFactory:
    def __init__(self, metrics: list[WSIMetric]) -> None:
        super().__init__()
        names = [metric.name for metric in metrics]
        if len(set(names)) != len(names):
            raise RuntimeError("Each individual metric must have a different name.")

        self._metrics = metrics

    @classmethod
    def for_segmentation(cls, *args, **kwargs):
        dices = WSIDiceMetric(*args, **kwargs)
        return cls([dices])

    @classmethod
    def for_wsi_classification(cls, config):
        raise NotImplementedError

    @classmethod
    def for_tile_classification(cls, roi_name: str, label: str, threshold: float):
        raise NotImplementedError

    def process_batch(
        self, predictions: torch.Tensor, target: torch.Tensor, wsi_name: str, roi: torch.Tensor | None
    ) -> None:
        for metric in self._metrics:
            metric.process_batch(predictions, target, wsi_name=wsi_name, roi=roi)

    def get_average_score(self) -> dict[str, float]:
        output = {}
        for metric in self._metrics:
            output.update(metric.get_average_score())
        return output

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

    def __repr__(self):
        return f"{type(self).__name__}(metrics={self._metrics})"


def _get_intersection_and_cardinality(
    predictions: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None, num_classes: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:

    soft_predictions = F.softmax(predictions, dim=1)
    corrected_roi = None
    if roi is not None:
        # Correct the ROI to remove any unannotated regions (see github issue #11)
        background = (target[:, 0, :, :] == 1).int()
        corrected_roi = roi.squeeze(1) - background
        # Clip the values below 0 after the subtraction.
        corrected_roi[corrected_roi < 0] = 0

    predictions = soft_predictions.argmax(dim=1)
    _target = target.argmax(dim=1)

    dice_components = []
    for class_idx in range(num_classes):
        curr_predictions = (predictions == class_idx).int()
        curr_target = (_target == class_idx).int()
        # Compute the dice score
        if corrected_roi is not None:
            intersection = torch.sum((curr_predictions * curr_target) * corrected_roi, dim=(0, 1, 2))
            cardinality = torch.sum(curr_predictions * corrected_roi, dim=(0, 1, 2)) + torch.sum(
                curr_target * corrected_roi, dim=(0, 1, 2)
            )
        else:
            intersection = torch.sum((curr_predictions * curr_target), dim=(0, 1, 2))
            cardinality = torch.sum(curr_predictions, dim=(0, 1, 2)) + torch.sum(curr_target, dim=(0, 1, 2))
        dice_components.append((intersection, cardinality))
    return dice_components


def _compute_dice(intersection: torch.Tensor, cardinality: torch.Tensor) -> torch.Tensor:
    dice_score = 2.0 * intersection / cardinality
    dice_score[dice_score.isnan()] = 1.0
    return dice_score
