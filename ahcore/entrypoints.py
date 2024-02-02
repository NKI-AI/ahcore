"""
Entrypoints
"""
from __future__ import annotations

import os
import pathlib
from pprint import pformat

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import Logger
from torch import nn

from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger, log_hyperparameters

logger = get_logger(__name__)


def load_weights_file(model: LightningModule, config: DictConfig) -> LightningModule:
    """Load a model from a checkpoint file.

    Parameters
    ----------
    model: LightningModule
        The model to load the weights into.
    config : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    LightningModule
        The model loaded from the checkpoint file.
    """
    if config.task_name == "inference" or config.task_name == "train":
        # Load checkpoint weights
        lit_ckpt = torch.load(config.ckpt_path)
        model.load_state_dict(lit_ckpt["state_dict"], strict=True)

    return model


def create_datamodule(
        config: DictConfig,
) -> tuple[DataDescription, LightningDataModule]:
    # Load generic description of the data
    if not config.data_description.get("_target_"):
        raise NotImplementedError(f"No data description defined in <{config.data_description}>")
    data_description: DataDescription = hydra.utils.instantiate(config.data_description)

    if config.datamodule.get("_target_"):
        logger.info(f"Instantiating datamodule <{config.datamodule._target_}>")  # noqa
        if not config.pre_transform.get("_target_"):
            raise RuntimeError("No pre-transform defined in <config.pre_transform>")
        logger.info(f"Instantiating pre_transforms <{config.pre_transform._target_}>")  # noqa
        pre_transform = hydra.utils.instantiate(config.pre_transform, data_description=data_description)
        datamodule: LightningDataModule = hydra.utils.instantiate(
            config.datamodule,
            data_description=data_description,
            pre_transform=pre_transform,
        )
        logger.info(pformat(data_description))  # TODO: Use nice rich formatting
        return data_description, datamodule

    raise NotImplementedError(f"No datamodule target found in <{config.datamodule}>")


def train(config: DictConfig) -> torch.Tensor | None:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.
    Arguments
    ---------
    config : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    Optional : float
        Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # TODO: Configurable?
    torch.set_float32_matmul_precision("high")  # type: ignore

    # Convert relative ckpt path to absolute path if necessary
    checkpoint_path = config.get("ckpt_path")
    if checkpoint_path and not os.path.isabs(checkpoint_path):
        config.trainer.resume_from_checkpoint = pathlib.Path(hydra.utils.get_original_cwd()) / checkpoint_path

    data_description, datamodule = create_datamodule(config)

    # Init augmentations
    augmentations: dict[str, nn.Module] | None = None
    if "augmentations" in config:
        augmentations = {}
        for stage in config.augmentations:
            if not config.augmentations[stage].get("_target_"):
                raise NotImplementedError(f"No augmentations target found in <{config.augmentations[stage]}>")
            logger.info(f"Instantiating {stage} augmentations <{config.augmentations[stage]._target_}>")  # noqa
            augmentations[stage] = hydra.utils.instantiate(
                config.augmentations[stage],
                data_description=data_description,
                data_module=datamodule,
                _convert_="object",
            )

    if not config.losses.get("_target_"):
        raise NotImplementedError(f"No loss target found in <{config.metrics}>")
    loss = hydra.utils.instantiate(config.losses)

    metrics: dict[str, nn.Module] | None = None
    if "metrics" in config:
        metrics = {}
        for metric_class in config.metrics:
            if not config.metrics[metric_class].get("_target_"):
                raise NotImplementedError(f"No metrics target found in <{config.metrics[metric_class]}>")
            logger.info(f"Instantiating metrics <{config.metrics[metric_class]._target_}>")  # noqa
            metrics[metric_class] = hydra.utils.instantiate(
                config.metrics[metric_class], data_description=data_description
            )
    logger.info(f"Metrics: {metrics}")

    # Init lightning model
    if not config.lit_module.get("_target_"):
        raise NotImplementedError(f"No model target found in <{config.lit_module}>")

    logger.info(f"Instantiating model <{config.lit_module._target_}>")  # noqa
    model: LightningModule = hydra.utils.instantiate(
        config.lit_module,
        data_description=data_description,
        augmentations=augmentations,
        loss=loss,
        metrics=metrics,
        _convert_="partial",
    )

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                logger.info("Instantiating callback <%s>", cb_conf._target_)  # noqa
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    lightning_loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                logger.info("Instantiating logger <%s>", lg_conf._target_)  # noqa
                lightning_loggers.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    if config.trainer.get("_target_"):
        logger.info("Instantiating trainer <%s>", str(config.trainer._target_))  # noqa
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=lightning_loggers,
            _convert_="partial",
        )

    else:
        raise NotImplementedError(f"No trainer target found in <{config.trainer}>")

    # Send some parameters from config to all lightning loggers
    logger.info("Logging hyperparameters...")
    log_hyperparameters(config=config, model=model, trainer=trainer)

    if config.get("train"):
        logger.info("Starting training...")
        trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found. "
            "Make sure the `optimized_metric` in `hparams_search` config is correct."
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test_after_training"):
        checkpoint_path = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            checkpoint_path = None
        logger.info("Starting testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

    # Make sure everything closed properly
    logger.info("Finalizing...")

    # Print path to best checkpoint
    if trainer.checkpoint_callback:
        if not config.trainer.get("fast_dev_run") and config.get("train"):
            logger.info(f"Best model checkpoint at {trainer.checkpoint_callback.best_model_path}")  # type: ignore

    # Return metric score for hyperparameter optimization
    return score


def inference(config: DictConfig) -> None:
    """Contains the inference pipeline.
    Arguments
    ---------
    config : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    checkpoint_path = config.get("ckpt_path")
    if not checkpoint_path:
        raise RuntimeError("No checkpoint inputted in config.ckpt_path")
    if checkpoint_path and not os.path.isabs(checkpoint_path):
        config.trainer.resume_from_checkpoint = pathlib.Path(hydra.utils.get_original_cwd()) / checkpoint_path

    data_description, datamodule = create_datamodule(config)

    # Init augmentations
    augmentations: dict[str, nn.Module] | None = None
    if "augmentations" in config:
        augmentations = {}
        for stage in config.augmentations:
            if not config.augmentations[stage].get("_target_"):
                raise NotImplementedError(f"No augmentations target found in <{config.augmentations[stage]}>")
            logger.info(f"Instantiating {stage} augmentations <{config.augmentations[stage]._target_}>")  # noqa
            augmentations[stage] = hydra.utils.instantiate(
                config.augmentations[stage],
                data_description=data_description,
                data_module=datamodule,
                _convert_="object",
            )
    # Init lightning model
    if not config.lit_module.get("_target_"):
        raise NotImplementedError(f"No model target found in <{config.lit_module}>")
    logger.info(f"Instantiating model <{config.lit_module._target_}>")  # noqa

    if config.task_name == "extract_features":
        config.lit_module.model.weights_path = config.ckpt_path
    model: LightningModule = hydra.utils.instantiate(
        config.lit_module,
        augmentations=augmentations,
        data_description=data_description,
        _convert_="partial",
    )

    # Load checkpoint weights
    model = load_weights_file(model, config)

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                logger.info("Instantiating callback <%s>", cb_conf._target_)  # noqa
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    lightning_loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                logger.info("Instantiating logger <%s>", lg_conf._target_)  # noqa
                lightning_loggers.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    if config.trainer.get("_target_"):
        logger.info("Instantiating trainer <%s>", str(config.trainer._target_))
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=lightning_loggers,
            _convert_="partial",
        )
    else:
        raise NotImplementedError(f"No trainer target found in <{config.trainer}>")

    # Inference
    logger.info("Starting inference...")
    trainer.predict(model=model, datamodule=datamodule, return_predictions=False)

    # Make sure everything closed properly
    logger.info("Finalizing...")
