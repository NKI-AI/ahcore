"""
This module contains the core Lightning module for ahcore. This module is responsible for:
- Training, Validation and Inference
- Wrapping models
"""

from __future__ import annotations

import functools
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import TrainerFn
from torch import nn

from ahcore.exceptions import ConfigurationError
from ahcore.metrics import MetricFactory, WSIMetricFactory
from ahcore.models.base_jit_model import BaseAhcoreJitModel
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
        model: nn.Module | BaseAhcoreJitModel | functools.partial,
        optimizer: torch.optim.optimzer.Optimizer,  # noqa
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
        if isinstance(model, BaseAhcoreJitModel):
            self._model = model
        elif isinstance(model, functools.partial):
            try:
                self._num_classes = data_description.num_classes
            except AttributeError:
                raise AttributeError("num_classes must be specified in data_description")
            self._model = model(out_channels=self._num_classes)
        elif isinstance(model, nn.Module):
            self._model = model
        else:
            raise TypeError(f"The class of models: {model.__class__} is not supported on ahcore")
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
        self._validation_counter = 0

    def on_train_epoch_start(self) -> None:
        # Reset the validation run counter at the start of each training epoch
        self._validation_counter = 0

    def on_validation_end(self) -> None:
        # Increment the counter each time validation starts
        self._validation_counter += 1

    @property
    def validation_counter(self) -> int:
        return self._validation_counter

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
        return output

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self._augmentations and "predict" in self._augmentations:
            batch = self._augmentations["predict"](batch)

        _relevant_dict = {k: v for k, v in batch.items() if k in self.RELEVANT_KEYS}
        batch = {**batch, **self._get_inference_prediction(batch["image"])}
        _prediction = batch["prediction"]
        output = {"prediction": _prediction, **_relevant_dict}
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
