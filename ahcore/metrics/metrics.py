"""
Metrics module, including factory.
"""
from __future__ import annotations

import abc
from collections import defaultdict
from typing import Any, List, Tuple

import kornia as K
import torch
import torch.nn.functional as F  # noqa

from ahcore.exceptions import ConfigurationError
from ahcore.utils.data import DataDescription


class TileMetric:
    def __init__(self, data_description: DataDescription) -> None:
        """Initialize the metric class"""
        self._data_description = data_description
        self.name: str | None = None

    @abc.abstractmethod
    def __call__(
        self, predictions: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Call metric computation"""


class DetectionMetric(TileMetric):
    def __init__(self, data_description: DataDescription) -> None:
        """
        Metric computing Recall, Precision and F1 over classes. The classes are derived from the index_map that's
        defined in the data_description.

        The `__call__` returns the recall, precision and f1 score for each class separate and averaged, with the class
        name (prefixed with fn_name/) as key
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
            raise ConfigurationError("`index_map` is required for to setup the classification metric.")
        else:
            _index_map = self._data_description.index_map

        if self._data_description.point_radius_microns is None or self._data_description.training_grid.mpp is None:
            raise ConfigurationError("`point_radius_microns` required for to setup the classification metric.")
        self._hit_criterion_radius = (
            self._data_description.point_radius_microns / self._data_description.training_grid.mpp
        )

        _label_to_class = {v: k for k, v in _index_map.items()}
        self._label_to_class = _label_to_class

    def __call__(
        self, predictions: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        # Distance for different classes and out of range is inf
        l2_dist = torch.cdist(predictions[:, :, :2], target[:, :, :2])
        diff_idx = predictions[:, :, 2:3] != target[:, :, 2:3].permute((0, 2, 1))
        oor_idx = (l2_dist > self._hit_criterion_radius).bool()
        l2_dist[diff_idx | oor_idx] = torch.inf

        # Create block diagonal with only valid distances
        l2_dist_block = torch.block_diag(*l2_dist)  # type: ignore
        l2_dist_block[~torch.block_diag(*torch.ones_like(l2_dist)).bool()] = torch.inf  # type: ignore

        # Find mutual nearest neighbors for valid predictions
        prediction_labels = predictions[:, :, 2].reshape(-1, 1)
        target_labels = target[:, :, 2].reshape(-1, 1)
        _, match_idxs = K.feature.match_mnn(prediction_labels, target_labels, l2_dist_block)
        match_labels_predictions = prediction_labels[match_idxs[:, 0]]
        match_labels_target = target_labels[match_idxs[:, 1]]
        assert all(match_labels_predictions == match_labels_target)

        # Calculate scores (assume background is always class 0 for now)
        confusion_matrix = torch.zeros((self._num_classes - 1, 3))
        for label_idx, label in self._label_to_class.items():
            if label == "background":
                continue
            tp = (match_labels_predictions == label_idx).sum()
            fp = (prediction_labels == label_idx).sum() - tp
            fn = (target_labels == label_idx).sum() - tp

            # Assume background is always class 0 for now
            confusion_matrix[label_idx - 1] += torch.Tensor([tp, fp, fn])

        # Caculate metrics per class and average
        metrics_dict = self._calculate_metrics_from_confusion_matrix(confusion_matrix)
        output = {
            f"{metric_name}/{self._label_to_class[idx]}": metric_values[idx - 1]
            for metric_name, metric_values in metrics_dict.items()
            for idx in range(1, self._num_classes)
        }
        output.update(
            {f"{metric_name}/all": torch.mean(metric_values) for metric_name, metric_values in metrics_dict.items()}
        )
        return output

    def _calculate_metrics_from_confusion_matrix(
        self, confusion_matrix: torch.Tensor, epsilon: float = 1e-12
    ) -> dict[str, torch.Tensor]:
        tp, fp, fn = confusion_matrix[:, 0], confusion_matrix[:, 1], confusion_matrix[:, 2]
        precision = tp / torch.clamp((tp + fp), min=epsilon)
        recall = tp / torch.clamp((tp + fn), min=epsilon)
        f1 = 2 * (precision * recall) / torch.clamp((precision + recall), min=epsilon)
        return {"precision": precision, "recall": recall, "f1": f1}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_classes={self._num_classes})"


class DiceMetric(TileMetric):
    def __init__(self, data_description: DataDescription) -> None:
        r"""
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

    def __call__(
        self, predictions: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        dice_components = _get_intersection_and_cardinality(predictions, target, roi, self._num_classes)
        dices = []
        for intersection, cardinality in dice_components:
            # dice_score is a float
            dice_score = _compute_dice(intersection, cardinality)
            dices.append(dice_score)

        output = {f"{self.name}/{self._label_to_class[idx]}": dices[idx] for idx in range(0, self._num_classes)}
        return output

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_classes={self._num_classes})"


class MetricFactory:
    # TODO: this should be rewritten to actually be a factory
    """Factory to create the metrics. These are fixed for the different tasks
    (e.g., segmentation, detection, whole-slide-level classification.
    """

    def __init__(self, metrics: list[TileMetric]) -> None:
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
    def for_detection(cls, *args: Any, **kwargs: Any) -> MetricFactory:
        detections = DetectionMetric(*args, **kwargs)
        return cls([detections])

    @classmethod
    def for_segmentation(cls, *args: Any, **kwargs: Any) -> MetricFactory:
        dices = DiceMetric(*args, **kwargs)
        return cls([dices])

    @classmethod
    def for_wsi_classification(cls, *args: Any, **kwargs: Any) -> MetricFactory:
        raise NotImplementedError

    @classmethod
    def for_tile_classification(cls, *args: Any, **kwargs: Any) -> MetricFactory:
        raise NotImplementedError

    def __call__(
        self, predictions: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        output = {}
        for metric in self._metrics:
            output.update(metric(predictions, target, roi=roi))
        return output

    def __repr__(self) -> str:
        return f"{type(self).__name__}(metrics={self._metrics})"


class WSIMetric(abc.ABC):
    def __init__(self, data_description: DataDescription) -> None:
        """Initialize the WSI metric class"""
        self.wsis: dict[str, Any] = {}
        self._data_description = data_description

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    # TODO: Fix Any
    def process_batch(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    # TODO: Fix Any
    def get_wsi_score(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def get_average_score(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class WSIDiceMetric(WSIMetric):
    """WSI Dice metric class, computes the dice score over the whole WSI"""

    def __init__(self, data_description: DataDescription, compute_overall_dice: bool = False) -> None:
        super().__init__(data_description=data_description)
        self.compute_overall_dice = compute_overall_dice
        self._num_classes = self._data_description.num_classes
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = "cpu"

        # Invert the index map
        _index_map = {}
        if self._data_description.index_map is None:
            raise ConfigurationError("`index_map` is required for to setup the wsi-dice metric.")
        else:
            _index_map = self._data_description.index_map

        _label_to_class: dict[int, str] = {v: k for k, v in _index_map.items()}
        _label_to_class[0] = "background"
        self._label_to_class = _label_to_class

    @property
    def name(self) -> str:
        return "wsi_dice"

    def process_batch(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor | None,
        wsi_name: str,
    ) -> None:
        if wsi_name not in self.wsis:
            self._initialize_wsi_dict(wsi_name)
        dice_components = _get_intersection_and_cardinality(
            predictions.to(self._device),
            target.to(self._device),
            roi.to(self._device) if roi is not None else None,
            self._num_classes,
        )
        for class_idx, (intersection, cardinality) in enumerate(dice_components):
            self.wsis[wsi_name][class_idx]["intersection"] += intersection
            self.wsis[wsi_name][class_idx]["cardinality"] += cardinality

    def get_wsi_score(self, wsi_name: str) -> None:
        for class_idx in self.wsis[wsi_name]:
            intersection = self.wsis[wsi_name][class_idx]["intersection"]
            cardinality = self.wsis[wsi_name][class_idx]["cardinality"]
            self.wsis[wsi_name][class_idx]["wsi_dice"] = _compute_dice(intersection, cardinality)

    def _get_overall_dice(self) -> dict[int, float]:
        """
        Compute the overall dice score (per class) over all the WSIs

        Returns
        -------
        dict
            Dictionary with the overall dice scores across wsis per class
        """
        overall_dices: dict[int, dict[str, float]] = {
            class_idx: {
                "total_intersection": 0.0,
                "total_cardinality": 0.0,
                "overall_dice": 0.0,
            }
            for class_idx in range(self._num_classes)
        }
        for wsi_name in self.wsis:
            for class_idx in range(self._num_classes):
                overall_dices[class_idx]["total_intersection"] += self.wsis[wsi_name][class_idx]["intersection"]
                overall_dices[class_idx]["total_cardinality"] += self.wsis[wsi_name][class_idx]["cardinality"]
        for class_idx in overall_dices.keys():
            intersection = overall_dices[class_idx]["total_intersection"]
            cardinality = overall_dices[class_idx]["total_cardinality"]
            overall_dices[class_idx]["overall_dice"] = (2 * intersection + 0.01) / (cardinality + 0.01)
        return {
            class_idx: torch.tensor(overall_dices[class_idx]["overall_dice"]).item()
            for class_idx in overall_dices.keys()
        }

    def _get_dice_averaged_over_total_wsis(self) -> dict[int, float]:
        """
        Compute the dice score (per class) averaged over all the WSIs

        Returns
        -------
        dict
            Dictionary with the dice scores averaged over all the WSIs per class
        """
        dices: dict[int, list[float]] = {class_idx: [] for class_idx in range(self._num_classes)}
        for wsi_name in self.wsis:
            self.get_wsi_score(wsi_name)
            for class_idx in range(self._num_classes):
                dices[class_idx].append(self.wsis[wsi_name][class_idx]["dice"].item())
        return {class_idx: sum(dices[class_idx]) / len(dices[class_idx]) for class_idx in dices.keys()}

    def _initialize_wsi_dict(self, wsi_name: str) -> None:
        self.wsis[wsi_name] = {
            class_idx: {"intersection": 0, "cardinality": 0, "dice": None} for class_idx in range(self._num_classes)
        }

    def get_average_score(
        self, precomputed_output: list[list[dict[str, dict[str, float]]]] | None = None
    ) -> dict[Any, Any]:
        if (
            precomputed_output is not None
        ):  # Used for multiprocessing, where multiple instances of this class are created
            avg_dict = self.static_average_wsi_dice(precomputed_output)
            if (
                avg_dict
            ):  # check if the precomputed output contained wsi dice scores, otherwise continue to compute it normally
                return avg_dict
        if self.compute_overall_dice:
            dices = self._get_overall_dice()
        else:
            dices = self._get_dice_averaged_over_total_wsis()
        avg_dict = {f"{self.name}/{self._label_to_class[idx]}": value for idx, value in dices.items()}
        return avg_dict

    @staticmethod
    def static_average_wsi_dice(precomputed_output: list[list[dict[str, dict[str, float]]]]) -> dict[str, float]:
        """Static method to compute the average WSI dice score over a list of WSI dice scores,
        useful for multiprocessing."""
        # Initialize defaultdicts to handle the sum and count of dice scores for each class
        class_sum: dict[str, float] = defaultdict(float)
        class_count: dict[str, int] = defaultdict(int)

        # Flatten the list and extract 'wsi_dice' dictionaries
        wsi_dices: list[dict[str, float]] = [
            wsi_metric.get("wsi_dice", {}) for sublist in precomputed_output for wsi_metric in sublist
        ]
        # Check if the list is empty -- then the precomputed output did not contain any wsi dice scores
        if not wsi_dices:
            return {}

        # Update sum and count for each class in a single pass
        for wsi_dice in wsi_dices:
            for class_name, dice_score in wsi_dice.items():
                class_sum[class_name] += dice_score
                class_count[class_name] += 1

        # Compute average dice scores in a dictionary comprehension with consistent naming
        avg_dice_scores = {
            f"{'wsi_dice'}/{class_name}": class_sum[class_name] / class_count[class_name]
            for class_name in class_sum.keys()
        }
        return avg_dice_scores

    def reset(self) -> None:
        self.wsis = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_classes={self._num_classes})"


class WSIMetricFactory:
    # TODO: this should be rewritten to actually be a factory
    def __init__(self, metrics: list[WSIMetric]) -> None:
        super().__init__()
        names = [metric.name for metric in metrics]
        if len(set(names)) != len(names):
            raise RuntimeError("Each individual metric must have a different name.")

        self._metrics = metrics

    @classmethod
    def for_segmentation(cls, *args: Any, **kwargs: Any) -> WSIMetricFactory:
        dices = WSIDiceMetric(*args, **kwargs)
        return cls([dices])

    @classmethod
    def for_wsi_classification(cls, *args: Any, **kwargs: Any) -> WSIMetricFactory:
        raise NotImplementedError

    @classmethod
    def for_tile_classification(cls, roi_name: str, label: str, threshold: float) -> WSIMetricFactory:
        raise NotImplementedError

    def process_batch(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        wsi_name: str,
        roi: torch.Tensor | None,
    ) -> None:
        for metric in self._metrics:
            metric.process_batch(predictions, target, wsi_name=wsi_name, roi=roi)

    def get_average_score(
        self, precomputed_output: list[list[dict[str, dict[str, float]]]] | None = None
    ) -> dict[str, float]:
        output = {}
        for metric in self._metrics:
            output.update(metric.get_average_score(precomputed_output))
        return output

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(metrics={self._metrics})"


def _get_intersection_and_cardinality(
    predictions: torch.Tensor,
    target: torch.Tensor,
    roi: torch.Tensor | None,
    num_classes: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    soft_predictions = F.softmax(predictions, dim=1)
    # if roi is not None:
    #     soft_predictions = soft_predictions * roi
    #     target = target * roi

    predictions = soft_predictions.argmax(dim=1)
    _target = target.argmax(dim=1)

    dice_components = []
    for class_idx in range(num_classes):
        curr_predictions = (predictions == class_idx).int()
        curr_target = (_target == class_idx).int()
        # Compute the dice score
        if roi is not None:
            intersection = torch.sum((curr_predictions * curr_target) * roi.squeeze(1), dim=(0, 1, 2))
            cardinality = torch.sum(curr_predictions * roi.squeeze(1), dim=(0, 1, 2)) + torch.sum(
                curr_target * roi.squeeze(1), dim=(0, 1, 2)
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
