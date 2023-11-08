"""
This module contains the core Lightning module for ahcore. This module is responsible for:
- Training, Validation and Inference
- Wrapping models
"""
from __future__ import annotations

from typing import Any, Optional

import kornia as K
import numpy as np
import pytorch_lightning as pl
import torch.optim.optimizer
from pytorch_lightning.trainer.states import TrainerFn
from skimage import measure
from torch import nn

from ahcore.exceptions import ConfigurationError
from ahcore.metrics import MetricFactory, WSIMetricFactory
from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger
from ahcore.utils.types import DlupDatasetSample

logger = get_logger(__name__)

LitModuleSample = dict[str, Any]  # TODO: This can be a TypedDict


class AhCoreLightningModule(pl.LightningModule):
    RELEVANT_KEYS = [
        "coordinates",
        "mpp",
        "path",
        "region_index",
        "grid_local_coordinates",
        "grid_index",
    ]

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,  # noqa
        data_description: DataDescription,
        loss: nn.Module | None = None,
        augmentations: dict[str, nn.Module] | None = None,
        metrics: dict[str, MetricFactory | WSIMetricFactory] | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,  # noqa
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "model",
                "augmentations",
                "metrics",
                "data_description",
                "loss",
            ],
        )  # TODO: we should send the hyperparams to the logger elsewhere

        self._num_classes = data_description.num_classes
        self._model = model(out_channels=self._num_classes)
        self._augmentations = augmentations

        self._loss = loss
        if metrics is not None:
            tile_metric = metrics.get("tile_level")
            wsi_metric = metrics.get("wsi_level", None)
            if tile_metric is not None and not isinstance(tile_metric, MetricFactory):
                raise ConfigurationError("Tile metrics must be of type MetricFactory")
            if wsi_metric is not None and not isinstance(wsi_metric, WSIMetricFactory):
                raise ConfigurationError("WSI metrics must be of type WSIMetricFactory")

            self._tile_metric = tile_metric
            self._wsi_metrics = wsi_metric

        self._data_description = data_description

    @property
    def wsi_metrics(self) -> WSIMetricFactory | None:
        return self._wsi_metrics

    @property
    def name(self) -> str:
        return str(self._model.__class__.__name__)

    def forward(self, sample: torch.Tensor) -> Any:
        """This function is only used during inference"""
        self._model.eval()
        return self._model.forward(sample)

    @property
    def data_description(self) -> DataDescription:
        return self._data_description

    def _compute_metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor | None,
        stage: TrainerFn | str,
    ) -> dict[str, torch.Tensor]:
        if not self._tile_metric:
            return {}

        _stage = stage.value if isinstance(stage, TrainerFn) else stage
        metrics = {f"{_stage}/{k}": v for k, v in self._tile_metric(prediction, target, roi).items()}
        return metrics

    def do_step(self, batch: DlupDatasetSample, batch_idx: int, stage: TrainerFn | str) -> LitModuleSample:
        if self._augmentations and stage in self._augmentations:
            batch = self._augmentations[stage](batch)

        if self._loss is None:
            raise RuntimeError(
                f"Loss is not defined for {self.__class__.__name__}. "
                f"This is required during training and validation"
            )

        _target = batch["target"]
        # Batch size is required for accurate loss calculation and logging
        batch_size = batch["image"].shape[0]
        # ROIs can reduce the usable area of the inputs, the loss should be scaled appropriately
        roi = batch.get("roi", None)

        if stage == "fit":
            _prediction = self._model(batch["image"])
            batch["prediction"] = _prediction
        else:
            batch = {**batch, **self._get_inference_prediction(batch["image"])}
            _prediction = batch["prediction"]

        loss = self._loss(_prediction, _target, roi)

        # The relevant_dict contains values to know where the tiles originate.
        _relevant_dict = {k: v for k, v in batch.items() if k in self.RELEVANT_KEYS}
        _metrics = self._compute_metrics(_prediction, _target, roi, stage=stage)
        _loss = loss.mean()
        # TODO: This can be a TypedDict
        output = {
            "loss": _loss,
            "loss_per_sample": loss.clone().detach(),
            "metrics": _metrics,
            **_relevant_dict,
        }
        if stage != "fit":
            output["prediction"] = _prediction

        _stage = stage.value if isinstance(stage, TrainerFn) else stage

        self.log(
            f"{_stage}/loss",
            _loss,
            batch_size=batch_size,
            sync_dist=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Log the metrics
        self.log_dict(
            _metrics,
            batch_size=batch_size,
            sync_dist=True,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        return output

    def _get_inference_prediction(self, _input: torch.Tensor) -> dict[str, torch.Tensor]:
        output = {}
        output["prediction"] = self._model(_input)
        return output

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        output = self.do_step(batch, batch_idx, stage="fit")
        return output

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        output = self.do_step(batch, batch_idx, stage="validate")

        # This is a sanity check. We expect the filenames to be constant across the batch.
        filename = batch["path"][0]
        if any([filename != f for f in batch["path"]]):
            raise ValueError("Filenames are not constant across the batch.")
        return output

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self._augmentations and "predict" in self._augmentations:
            batch = self._augmentations["predict"](batch)

        _relevant_dict = {k: v for k, v in batch.items() if k in self.RELEVANT_KEYS}
        batch = {**batch, **self._get_inference_prediction(batch["image"])}
        _prediction = batch["prediction"]
        output = {"prediction": _prediction, **_relevant_dict}

        # This is a sanity check. We expect the filenames to be constant across the batch.
        filename = batch["path"][0]
        if any([filename != f for f in batch["path"]]):
            raise ValueError("Filenames are not constant across the batch.")
        return output

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())  # type: ignore
        if self.hparams.scheduler is not None:  # type: ignore
            scheduler = self.hparams.scheduler(optimizer=optimizer)  # type: ignore
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "validate/loss",
                    "interval": "epoch",
                    "frequency": self.trainer.check_val_every_n_epoch,
                },
            }
        return {"optimizer": optimizer}


def _process_point_predictions(
    predictions: torch.Tensor,
    roi: Optional[torch.Tensor] = None,
    kernel_size: Optional[tuple[int, int] | int] = (5, 5),
    sigma: Optional[tuple[float, float]] = (1.0, 1.0),
    min_threshold: Optional[float] = 0.5,
) -> dict[str, torch.Tensor]:
    """Post-process segementation maps into point annotations.

    Model output is converted into point annotations with labels and confidences of the extracted region. This is done
    by applying a softmax/sigmoid on the predictions and optionally Gaussian blur and thresholding. For each pixel in
    the max and argmax will be viewed as the confidence and prediction respectively. Using `label` and `region_props`
    from `skimage.measure` regions will be extracted. The centroid of each region is considered a point prediction with
    confidence taken as the mean intensity of the region and label as the corresponding label in the argmax.

    Arguments
    ---------
    predictions: torch.Tensor
        Model output of shape `(N, C, H, W)`.
    roi: torch.Tensor
        ROI of shape `(N, 1, H, W)`
    kernel_size: tuple[int, int], optional
        Kernel size for 2D Gaussian blur. If `kernel_size=None` no blur will be applied.
        Default: (5, 5)
    sigma: tuple[float, float], optional
        Standard deviation for 2D Gaussian blur. If `sigma=None` no blur will be applied.
        Default: (1.0, 1.0)
    min_threshold: float, optional
        Minimum threshold for predictions. Values lower than threshold after the activation function will be set to 0.
        Default: 0.5

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary mapping `points`, `point_labels` and `point_confidences` to output as padded tensors of shape
        `(N, K, 2)`, `(N, K, 1)` and `(N, K, 1)` respectively. `K` is the largest number of points.
    """
    _predictions = torch.softmax(predictions, dim=1) if predictions.shape[1] > 1 else torch.sigmoid(predictions)
    if roi is not None:
        _predictions *= roi
    if sigma is not None and kernel_size is not None:
        _predictions = K.filters.gaussian_blur2d(_predictions, kernel_size=kernel_size, sigma=sigma)
    if min_threshold is not None:
        _predictions[_predictions < min_threshold] = 0

    _predictions_max, _predictions_argmax = torch.max(_predictions, dim=1)
    _predictions_max = _predictions_max.detach().cpu().numpy()
    _predictions_argmax = _predictions_argmax.detach().cpu().numpy()

    # Is kornia.contrib.connected_components a good alternative? It does not do the same thing
    _point_predictions = []
    for _sample_max, _sample_argmax in zip(_predictions_max, _predictions_argmax):
        _labels = measure.label(_sample_argmax, background=0)  # type: ignore
        _regions = measure.regionprops(_labels, intensity_image=_sample_max)  # type: ignore

        _sample_points = []
        for region in _regions:
            y, x = np.round(region.centroid).astype(int)
            # Centroid is not always in region (think horse shoe) and label is instance label
            region_label = np.unique(_sample_argmax[_labels == region.label])[0]
            region_confidence = region.intensity_mean
            _sample_points.append([x, y, region_label, region_confidence])

        # Sort predictions by confidences
        if _sample_points:
            _point_predictions.append(torch.tensor(sorted(_sample_points, key=lambda x: x[3], reverse=True)))
        else:
            _point_predictions.append(torch.empty((0, 4)))

    _padded_point_predictions = torch.nn.utils.rnn.pad_sequence(
        _point_predictions, batch_first=True, padding_value=torch.nan
    ).float()
    output = {
        "points": _padded_point_predictions[:, :, :2],
        "points_labels": _padded_point_predictions[:, :, 2],
        "points_confidences": _padded_point_predictions[:, :, 3],
    }
    return output
